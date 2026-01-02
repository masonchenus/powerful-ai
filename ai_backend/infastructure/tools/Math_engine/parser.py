import symengine as se  # Faster C++ version of SymPy
import re

class MathParser:
    def __init__(self):
        # Support for your massive 18-quintillion scale constants
        self.constants = {"pi": se.pi, "e": se.exp(1)}

    def sanitize_input(self, expression_str):
        """Removes dangerous characters to prevent code injection."""
        return re.sub(r'[^0-9a-zA-Z\s\+\-\*\/\^\(\)\.]', '', expression_str)

    def parse_to_symbolic(self, raw_expression):
        """
        Converts '10^243 - 2^pi' into a SymEngine object.
        Keeps numbers in 'Symbolic Form' so precision isn't lost.
        """
        clean_expr = self.sanitize_input(raw_expression)
        
        # Convert string to Symbolic Tree
        # This handles the logic, not the digits.
        expr = se.sympify(clean_expr, locals=self.constants)
        
        return expr

    def analyze_complexity(self, expr):
        """Decides if we need the GPU (Kernels) or just the CPU (BigNum)."""
        if se.count_ops(expr) > 1000 or "vector" in str(expr):
            return "GPU_VECTOR"
        return "CPU_BIG_NUM"