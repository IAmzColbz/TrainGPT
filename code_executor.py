# code_executor.py
# This file might be less critical if self-play is removed,
# but can be used for evaluating the model's generated code against ground truth
# outside the training loop (e.g., during a final test phase or for examples).
# For now, keeping it minimal.

import ast
import signal
from contextlib import contextmanager
import config
from utils import is_safe_code, clean_output
import logging

logger = logging.getLogger(__name__)

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    if hasattr(signal, 'SIGALRM'):
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
    else:
        logger.warning("SIGALRM not available. Execution timeout will not be enforced by signal.")
        original_handler = None
    try:
        yield
    finally:
        if hasattr(signal, 'SIGALRM') and original_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

class CodeExecutor:
    def execute_function_safely(self, program_str: str, function_name: str, args_tuple_str: str):
        """
        Executes a specific function from program_str with given arguments.
        Args:
            program_str: The Python code string defining the function.
            function_name: The name of the function to call (e.g., 'f').
            args_tuple_str: A string representation of the arguments tuple, e.g., "('hello', 10)".
        Returns:
            (True, result, None) if successful.
            (False, None, error_message) if failed.
        """
        if not isinstance(program_str, str) or not isinstance(args_tuple_str, str):
            return False, None, "Program and arguments must be strings."

        if not is_safe_code(program_str, config.FORBIDDEN_MODULES):
            return False, None, "Code contains forbidden modules or is unsafe."

        try:
            ast.parse(program_str)
        except SyntaxError as e:
            return False, None, f"Syntax error in program: {e}"
        except RecursionError:
            return False, None, "AST parsing recursion limit."

        safe_builtins = {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in ['eval', 'exec', 'open', 'compile', 'input', '__import__', 'exit', 'quit', 'help', 'globals', 'locals', 'vars']
        }
        execution_globals = {"__builtins__": safe_builtins}
        execution_locals = {}

        try:
            with time_limit(config.EXECUTION_TIMEOUT_SECONDS):
                exec(program_str, execution_globals, execution_locals)

                if function_name not in execution_locals or not callable(execution_locals[function_name]):
                    return False, None, f"Program does not define callable function '{function_name}'."

                # Parse arguments string into actual Python objects
                # This is a critical step for safety and correctness.
                # ast.literal_eval is safer than full eval.
                try:
                    args = ast.literal_eval(args_tuple_str)
                    if not isinstance(args, tuple): # Ensure it's a tuple for *args
                        args = (args,)
                except (ValueError, SyntaxError, TypeError) as e_args:
                    return False, None, f"Invalid argument format '{args_tuple_str}': {e_args}. Must be a literal tuple string e.g. \"('val1', 2)\"."

                # Call the function
                result = execution_locals[function_name](*args)
                return True, clean_output(result), None
        except TimeoutError:
            return False, None, "Execution timed out."
        except Exception as e_exec:
            return False, None, f"Runtime error during execution: {e_exec}"

    def verify_solution_output(self, actual_output: str, expected_output: str):
        if actual_output is None and expected_output is None: return True
        if actual_output is None or expected_output is None: return False
        return clean_output(actual_output) == clean_output(expected_output)