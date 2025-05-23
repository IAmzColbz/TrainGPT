# --- START OF FILE code_executor.py ---

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
    
    # Check if SIGALRM is available (not on Windows)
    if hasattr(signal, 'SIGALRM'):
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds)) # alarm expects integer seconds
    else:
        logger.warning("SIGALRM not available on this system (e.g., Windows). Execution timeout will not be enforced by signal.")
        original_handler = None # Indicate no handler was set
    
    try:
        yield
    finally:
        if hasattr(signal, 'SIGALRM') and original_handler is not None: # Only try to reset if it was set
            signal.alarm(0) # Disable the alarm
            signal.signal(signal.SIGALRM, original_handler) # Restore original handler

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
        except RecursionError: # Handle deeply nested ASTs
            return False, None, "AST parsing recursion limit hit."


        # Create a restricted set of builtins
        # Allow common safe builtins, explicitly disallow dangerous ones.
        safe_builtins_dict = {
            'print': print, 'len': len, 'range': range, 'list': list, 'dict': dict,
            'str': str, 'int': int, 'float': float, 'bool': bool, 'tuple': tuple,
            'set': set, 'abs': abs, 'max': max, 'min': min, 'sum': sum, 'sorted': sorted,
            'zip': zip, 'enumerate': enumerate, 'map': map, 'filter': filter,
            'isinstance': isinstance, 'issubclass': issubclass, 'round': round,
            'pow': pow, 'divmod': divmod, 'AssertionError': AssertionError,
            'TypeError': TypeError, 'ValueError': ValueError, 'IndexError': IndexError,
            'KeyError': KeyError, 'Exception': Exception, 'ArithmeticError': ArithmeticError,
            # Allow None, True, False by default as they are part of language
        }
        # Create a __builtins__ object that only exposes these safe functions
        restricted_builtins = {'__builtins__': safe_builtins_dict}
        
        execution_globals = {"__builtins__": restricted_builtins}
        execution_locals = {}

        try:
            with time_limit(config.EXECUTION_TIMEOUT_SECONDS):
                # Execute the program string in the restricted environment
                exec(program_str, execution_globals, execution_locals)

                if function_name not in execution_locals or not callable(execution_locals[function_name]):
                    return False, None, f"Program does not define callable function '{function_name}'."

                # Parse arguments string into actual Python objects using ast.literal_eval
                try:
                    args = ast.literal_eval(args_tuple_str)
                    if not isinstance(args, tuple): # Ensure it's a tuple for *args
                        args = (args,) # Wrap single non-tuple argument in a tuple
                except (ValueError, SyntaxError, TypeError) as e_args:
                    return False, None, f"Invalid argument format '{args_tuple_str}': {e_args}. Must be a literal tuple string e.g. \"('val1', 2)\" or \"(1,)\"."

                # Call the function
                result = execution_locals[function_name](*args)
                return True, clean_output(result), None
                
        except TimeoutError:
            return False, None, "Execution timed out."
        except SyntaxError as e_syn: # Should be caught by ast.parse earlier, but as safeguard
             return False, None, f"Syntax error during exec: {e_syn}"
        except RecursionError: # Safeguard for runtime recursion
            return False, None, "Runtime recursion limit hit."
        except Exception as e_exec:
            # More detailed error reporting for debugging model generations
            import traceback
            tb_str = traceback.format_exc()
            logger.debug(f"Runtime error during execution: {e_exec}\nCode:\n{program_str}\nArgs: {args_tuple_str}\nTraceback:\n{tb_str}")
            return False, None, f"Runtime error: {type(e_exec).__name__}: {e_exec}"

    def verify_solution_output(self, actual_output: str, expected_output: str):
        """Compares cleaned actual and expected outputs."""
        # Handle None cases explicitly for clarity
        if actual_output is None and expected_output is None: 
            return True
        if actual_output is None or expected_output is None: 
            # One is None, the other is not, so they are different
            return False
            
        # Both are not None, compare their cleaned string representations
        return clean_output(str(actual_output)) == clean_output(str(expected_output))