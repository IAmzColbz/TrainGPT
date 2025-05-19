# utils.py
import ast
import re

def extract_python_code_from_markdown(markdown_text: str) -> str:
    """
    Extracts Python code from a markdown string.
    Handles ```python ... ``` and ``` ... ``` blocks.
    Returns the first block found, or the original text if no block is found.
    """
    if not isinstance(markdown_text, str):
        return ""
    match = re.search(r"```(?:python\n)?(.*?)```", markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_text.strip()

def is_safe_code(code_string, forbidden_modules):
    if not isinstance(code_string, str): return False
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.split('.')[0] in forbidden_modules:
                        return False
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and \
                   node.func.id in ['open', 'eval', 'exec', '__import__', 'compile', 'input']:
                    return False
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in forbidden_modules:
                        return False
    except SyntaxError:
        return False
    except RecursionError:
        return False
    return True

def clean_output(output):
    if isinstance(output, str):
        return output.strip()
    return str(output)

def program_is_syntactically_valid(program_str): # Could be used for validating generated code
    if not isinstance(program_str, str): return False
    try:
        ast.parse(program_str)
        return True
    except SyntaxError:
        return False
    except RecursionError:
        return False

# These are less relevant without self-play proposer/solver, but kept for potential future use
# or if model generates structured output that needs parsing.
def proposer_output_has_separator(generated_text_proposer):
    if not isinstance(generated_text_proposer, str): return False
    return "###" in generated_text_proposer

def solver_output_has_tags(generated_text_solver):
    if not isinstance(generated_text_solver, str): return False
    return "<think>" in generated_text_solver and \
           "</think>" in generated_text_solver and \
           "<answer>" in generated_text_solver and \
           "</answer>" in generated_text_solver

def parse_think_answer(response_str: str):
    if not isinstance(response_str, str):
        return None, None, False
    think_match = re.search(r"<think>(.*?)</think>", response_str, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
    has_both_tag_pairs = ("<think>" in response_str and "</think>" in response_str and \
                          "<answer>" in response_str and "</answer>" in response_str)
    if think_match and answer_match:
        thought = think_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return thought, answer, True
    elif has_both_tag_pairs:
        return None, None, False
    return None, None, False