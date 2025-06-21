import streamlit as st
import numpy as np
import math
from scipy import integrate
from sympy import sympify, lambdify, SympifyError

# --- Page Configuration ---
st.set_page_config(
    page_title="Scientific Calculator",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS for better button styling ---
st.markdown("""
<style>
/* Style for calculator buttons */
.stButton > button {
    font-size: 24px !important;
    font-weight: bold !important;
    height: 60px !important;
    border-radius: 8px !important;
    font-family: 'Arial', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
}

/* Animated display */
.calculator-display {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 20px !important;
    border-radius: 12px !important;
    font-size: 28px !important;
    font-family: 'Courier New', monospace !important;
    color: white !important;
    text-align: right !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
    min-height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    word-break: break-all !important;
}

/* Expandable sections animation */
.streamlit-expanderHeader {
    background-color: #f0f2f6 !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Custom styles for calculus buttons */
.integral-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.integral-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
}

.derivative-button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.derivative-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(240, 147, 251, 0.6) !important;
    background: linear-gradient(135deg, #e879f9 0%, #ec4899 100%) !important;
}

/* Apply custom styles to specific buttons */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def safe_eval(expr_str, local_dict):
    """
    Safely evaluates a string expression using a dictionary of allowed functions.
    Uses SymPy's lambdify for safe and efficient evaluation.
    """
    try:
        # 1. Sympify the expression to convert it into a SymPy object
        sympy_expr = sympify(expr_str)

        # 2. Get all symbols (variables) from the expression
        symbols = sympy_expr.free_symbols

        # 3. Create a callable function from the SymPy expression
        # This function will take the symbol values as arguments
        func = lambdify(symbols, sympy_expr, modules=[local_dict, 'numpy'])

        # 4. Prepare the arguments for the function
        # For a basic calculator, we expect no free variables, so this should be empty
        args = {str(s): local_dict.get(str(s), 0) for s in symbols}

        return func(**args)
    except (SympifyError, NameError, TypeError, SyntaxError) as e:
        st.error("❌ Invalid Expression")
        return None

def transform_display(expression):
    """Converts emoji operators to standard math symbols for display"""
    return (
        expression
        .replace('➕', '+')
        .replace('➖', '-')
        .replace('✖️', '×')
        .replace('➗', '÷')
    )

# --- UI Layout ---
st.title("🧮 Scientific Calculator")
st.markdown("Use the buttons below and write mathematical expression. For calculus, use the dedicated sections.")

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'expression' not in st.session_state:
    st.session_state.expression = ""

# --- Main Calculator Interface ---
# Dictionary of allowed functions and constants for safe evaluation
allowed_funcs = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "log": np.log10, "ln": np.log,
    "sqrt": np.sqrt, "exp": np.exp,
    "pi": np.pi, "e": np.e,
    "abs": np.abs,
    "factorial": math.factorial
}

# Display area with better styling
st.markdown("### Current Expression:")
display_expr = st.session_state.expression if st.session_state.expression else "0"
display_expr = transform_display(display_expr)  # Apply transformation
st.markdown(f"<div class='calculator-display'>{display_expr}</div>", unsafe_allow_html=True)

st.markdown("---")

# Calculator Buttons with uniform symbols
buttons = [
    ('7', '8', '9', '➗'),
    ('4', '5', '6', '✖️'),
    ('1', '2', '3', '➖'),
    ('0', '(', ')', '➕'),
    ('.', 'C', '⌫', '=')
]

# Updated key mapping
key_map = {
    '➕': 'plus', '➖': 'minus', '➗': 'divide', '✖️': 'multiply',
    '=': 'equals', '⌫': 'backspace', '.': 'dot', '(': 'open_paren',
    ')': 'close_paren', 'C': 'clear'
}

# Symbol mapping for calculation
symbol_map = {
    '➕': '+',
    '➖': '-',
    '➗': '/',
    '✖️': '*',
    '⌫': '<-'
}

for row in buttons:
    cols = st.columns(4)
    for i, label in enumerate(row):
        # Create a safe key for the button
        safe_key = f"btn_{key_map.get(label, label)}"

        if cols[i].button(label, use_container_width=True, key=safe_key):
            if label == '=':
                if st.session_state.expression:
                    # Replace symbols for calculation
                    eval_expr = st.session_state.expression
                    # Replace π with pi for calculation, factorial( is already correct
                    eval_expr = eval_expr.replace('π', 'pi')
                    # Replace emoji symbols with actual operators for evaluation
                    eval_expr = eval_expr.replace('➕', '+').replace('➖', '-')
                    eval_expr = eval_expr.replace('➗', '/').replace('✖️', '*').replace('^', '**')

                    result = safe_eval(eval_expr, allowed_funcs)
                    if result is not None:
                        st.session_state.history.append(f"{transform_display(st.session_state.expression)} = {result}")
                        st.session_state.expression = str(result)
                        st.rerun()
            elif label == 'C':
                st.session_state.expression = ""
                st.rerun()
            elif label == '⌫':
                st.session_state.expression = st.session_state.expression[:-1]
                st.rerun()
            else:
                st.session_state.expression += label
                st.rerun()

# --- Advanced Functions Sections ---
st.markdown("---")
st.header("🔬 Advanced Operations")

# --- Scientific Functions Expander ---
with st.expander("📐 Trigonometric & Logarithmic Functions"):
    func_buttons = [
        ('sin()', 'cos()', 'tan()'),
        ('asin()', 'acos()', 'atan()'),
        ('log()', 'ln()', 'sqrt()'),
        ('exp()', 'π', 'e'),
        ('abs()', 'n!', '^')
    ]
    for row in func_buttons:
        cols = st.columns(3)
        for i, label in enumerate(row):
            if cols[i].button(label, use_container_width=True, key=f"func_btn_{label}"):
                if label == 'π':
                    st.session_state.expression += 'π'
                elif label == 'n!':
                    st.session_state.expression += 'factorial('
                elif '()' in label:
                    insert_text = label[:-1]
                    st.session_state.expression += insert_text
                else:
                    st.session_state.expression += label
                st.rerun()

# --- Calculus Section ---
with st.expander("🧠 Calculus: Integration & Differentiation"):
    st.subheader("∫ Definite Integration")

    col_int1, col_int2 = st.columns(2)
    func_str_int = col_int1.text_input("Function f(x)", "x**2", help="Enter a function of x. Use ** for power.")
    variable_int = col_int2.text_input("Variable", "x", max_chars=1, help="The variable of integration.", key="var_int")

    col_lim1, col_lim2 = st.columns(2)
    lower_limit = col_lim1.number_input("Lower Limit (a)", value=0.0, format="%.4f")
    upper_limit = col_lim2.number_input("Upper Limit (b)", value=1.0, format="%.4f")

    if st.button("Calculate Integral", use_container_width=True, type="primary"):
        try:
            x = sympify(variable_int)
            sympy_func = sympify(func_str_int)
            f = lambdify(x, sympy_func, 'numpy')
            result, error = integrate.quad(f, lower_limit, upper_limit)

            st.success(f"✅ Result: {result}")
            st.info(f"📊 Estimated Error: {error}")
            st.session_state.history.append(f"∫({func_str_int}) from {lower_limit} to {upper_limit} = {result}")

        except (SympifyError, TypeError, ValueError) as e:
            st.error(f"❌ Error in integration: {e}")

    st.markdown("---")
    st.subheader("d/dx Differentiation")

    col_diff1, col_diff2 = st.columns(2)
    func_str_diff = col_diff1.text_input("Function f(x)", "sin(x)", help="Enter a function of x. Use ** for power.")
    variable_diff = col_diff2.text_input("Variable", "x", max_chars=1, help="The variable of differentiation.",
                                         key="var_diff")

    eval_point = st.number_input("Point (x)", value=0.0, format="%.4f",
                                 help="The point at which to evaluate the derivative.")

    if st.button("Calculate Derivative", use_container_width=True, type="primary"):
        try:
            x_sym = sympify(variable_diff)
            sympy_func = sympify(func_str_diff)

            derivative_expr = sympy_func.diff(x_sym)
            result = derivative_expr.subs(x_sym, eval_point).evalf()

            st.success(f"✅ Derivative at x={eval_point}: {result}")
            st.latex(f"\\frac{{d}}{{d{x_sym}}} \\left( {sympy_func} \\right) = {derivative_expr}")
            st.session_state.history.append(f"d/d{x_sym}({func_str_diff}) at {eval_point} = {result}")

        except (SympifyError, TypeError, ValueError) as e:
            st.error(f"❌ Error in differentiation: {e}")

# --- History Sidebar ---
with st.sidebar:
    st.header("📋 History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            st.text_area(f"#{len(st.session_state.history) - i}", value=entry, height=75, disabled=True,
                         key=f"hist_{i}")

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No calculations yet.")

# --- Instructions ---
st.markdown("---")
st.markdown("""
### 📝 Instructions:
- **Basic Operations**: Use the number pad and operators (➕, ➖, ✖️, ➗)
- **Scientific Functions**: Click the trigonometric functions to add them to your expression
- **Calculus**: Use the dedicated sections for integration and differentiation
- **History**: View your calculation history in the sidebar
- **Examples**: 
  - Basic: 5 + 3 = 8
  - Scientific: sin(π/2) = 1.0
  - Factorial: factorial(5) = 120
""")