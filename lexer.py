import enum
import sys
import numpy as np

# --- 1. LEXER (The Tokenizer) ---

class TokenType(enum.Enum):
    EOF = -1
    NEWLINE = 0
    NUMBER = 1
    IDENTIFIER = 2
    STRING = 3
    # Symbols
    SECTION_START = 10  # [
    SECTION_END   = 11  # ]
    EQ            = 12  # =
    LPAREN        = 13  # (
    RPAREN        = 14  # )
    COMMA         = 15  # ,

class Token:
    def __init__(self, text, kind):
        self.text = text
        self.kind = kind

    @staticmethod
    def output_kind(kind):
        return kind.name

class Lexer:
    def __init__(self, source):
        # We append a newline to ensure the last line is parsed correctly
        self.source = source + '\n'
        self.cur_char = ''
        self.cur_pos = -1
        self.next_char()

    def next_char(self):
        self.cur_pos += 1
        if self.cur_pos >= len(self.source):
            self.cur_char = '\0'  # EOF
        else:
            self.cur_char = self.source[self.cur_pos]

    def peek(self):
        if self.cur_pos + 1 >= len(self.source):
            return '\0'
        return self.source[self.cur_pos + 1]

    def skip_whitespace(self):
        # We skip spaces and tabs, but NEWLINES are significant tokens here
        while self.cur_char == ' ' or self.cur_char == '\t' or self.cur_char == '\r':
            self.next_char()

    def skip_comment(self):
        if self.cur_char == '#':
            while self.cur_char != '\n' and self.cur_char != '\0':
                self.next_char()

    def get_token(self):
        self.skip_whitespace()
        self.skip_comment()
        
        # Double check whitespace after comment skip
        self.skip_whitespace()

        token = None

        if self.cur_char == '\0':
            token = Token('', TokenType.EOF)
        elif self.cur_char == '\n':
            token = Token('\n', TokenType.NEWLINE)
            self.next_char()
        elif self.cur_char == '[':
            token = Token('[', TokenType.SECTION_START)
            self.next_char()
        elif self.cur_char == ']':
            token = Token(']', TokenType.SECTION_END)
            self.next_char()
        elif self.cur_char == '=':
            token = Token('=', TokenType.EQ)
            self.next_char()
        elif self.cur_char == '(':
            token = Token('(', TokenType.LPAREN)
            self.next_char()
        elif self.cur_char == ')':
            token = Token(')', TokenType.RPAREN)
            self.next_char()
        elif self.cur_char == ',':
            token = Token(',', TokenType.COMMA)
            self.next_char()
            
        elif self.cur_char.isdigit() or self.cur_char == '-' or self.cur_char == '.':
            # Number Parsing (Integers and Floats)
            start_pos = self.cur_pos
            while self.cur_char.isdigit() or self.cur_char == '.' or self.cur_char == '-':
                self.next_char()
            text = self.source[start_pos : self.cur_pos]
            token = Token(text, TokenType.NUMBER)
            
        elif self.cur_char.isalpha() or self.cur_char == '_':
            # Identifier Parsing
            start_pos = self.cur_pos
            while self.cur_char.isalnum() or self.cur_char == '_':
                self.next_char()
            text = self.source[start_pos : self.cur_pos]
            token = Token(text, TokenType.IDENTIFIER)
            
        else:
            # Unknown character (skip it)
            self.next_char()
            return self.get_token()

        return token

# --- 2. PARSER (The Grammar Logic) ---

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.cur_token = None
        self.peek_token = None
        self.next_token()
        self.next_token() # Call twice to initialize current and peek

        # The "Emitter" storage
        self.data = {
            'boundaries': [],
            'constraints': [],
            'nodes': [],
            'edges': []
        }
        self.current_section = None

    def check_token(self, kind):
        return self.cur_token.kind == kind

    def check_peek(self, kind):
        return self.peek_token.kind == kind

    def match(self, kind):
        if not self.check_token(kind):
            raise SyntaxError(f"Expected {kind.name}, got {self.cur_token.kind.name} ('{self.cur_token.text}')")
        self.next_token()

    def next_token(self):
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.get_token()

    def abort(self, message):
        sys.exit("Error: " + message)

    # --- GRAMMAR RULES ---

    def program(self):
        print("PROGRAM")
        
        # Handle empty newlines at start
        while self.check_token(TokenType.NEWLINE):
            self.next_token()

        while not self.check_token(TokenType.EOF):
            self.section()
            
        # Final Conversion to NumPy
        self.finalize_arrays()

    def section(self):
        # Grammar: [ HEADER ] NEWLINE statements
        self.match(TokenType.SECTION_START)
        
        if not self.check_token(TokenType.IDENTIFIER):
            self.abort("Expected section name identifier")
            
        header_name = self.cur_token.text
        self.current_section = header_name.lower()
        print(f"  SECTION: {header_name}")
        self.next_token() # Consume identifier
        
        self.match(TokenType.SECTION_END)
        self.nl() # Require newline after header

        # Parse statements until next section or EOF
        while not self.check_token(TokenType.EOF) and not self.check_token(TokenType.SECTION_START):
            if self.check_token(TokenType.NEWLINE):
                self.next_token()
                continue
            
            # Dispatch based on section type
            if self.current_section == 'constraints':
                self.statement_constraint()
            elif self.current_section in ['nodes', 'edges', 'boundaries']:
                self.statement_table_row()
            else:
                self.abort(f"Unknown section: {self.current_section}")

    def statement_constraint(self):
        # Grammar: ID = FUNC ( ARGS ) NEWLINE
        name = self.cur_token.text
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.EQ)
        
        func_name = self.cur_token.text
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.LPAREN)
        
        args = self.parse_args()
        
        self.match(TokenType.RPAREN)
        self.nl()

        # Emit (Store data)
        print(f"    --> Constraint: {name} = {func_name}({args})")
        self.data['constraints'].append({
            'name': name,
            'type': func_name,
            'params': args
        })

    def parse_args(self):
        # Grammar: ARG { , ARG }
        # ARG: key = value
        args = {}
        while not self.check_token(TokenType.RPAREN):
            key = self.cur_token.text
            self.match(TokenType.IDENTIFIER)
            self.match(TokenType.EQ)
            
            # Value can be NUMBER or TUPLE (LPAREN NUM COMMA NUM RPAREN)
            if self.check_token(TokenType.NUMBER):
                val = float(self.cur_token.text)
                self.match(TokenType.NUMBER)
            elif self.check_token(TokenType.LPAREN):
                # Tuple (x, y)
                self.match(TokenType.LPAREN)
                v1 = float(self.cur_token.text)
                self.match(TokenType.NUMBER)
                self.match(TokenType.COMMA)
                v2 = float(self.cur_token.text)
                self.match(TokenType.NUMBER)
                self.match(TokenType.RPAREN)
                val = (v1, v2)
            else:
                self.abort(f"Unexpected value type for argument {key}")
                
            args[key] = val

            if self.check_token(TokenType.COMMA):
                self.next_token()
                
        return args

    def statement_table_row(self):
        # Grammar: VAL { VAL } NEWLINE
        # Generic row parser for nodes/edges
        row = []
        while not self.check_token(TokenType.NEWLINE) and not self.check_token(TokenType.EOF):
            if self.check_token(TokenType.NUMBER):
                # Try integer first, then float
                txt = self.cur_token.text
                val = float(txt)
                if '.' not in txt: val = int(val)
                row.append(val)
                self.match(TokenType.NUMBER)
            elif self.check_token(TokenType.IDENTIFIER):
                row.append(self.cur_token.text)
                self.match(TokenType.IDENTIFIER)
            else:
                self.abort("Unexpected token in table row")

        if row: # Only emit non-empty rows
            self.data[self.current_section].append(row)
            
        self.nl()

    def nl(self):
        self.match(TokenType.NEWLINE)
        while self.check_token(TokenType.NEWLINE):
            self.next_token()

    def finalize_arrays(self):
        print("--> Compiling to NumPy Arrays...")
        # Convert Lists to Structured Arrays
        
        # Nodes
        if self.data['nodes']:
            # Assume: ID, X, Y, ConstraintID (Optional)
            # We handle ragged arrays by filling defaults
            clean_nodes = []
            for r in self.data['nodes']:
                nid, x, y = r[0], r[1], r[2]
                cid = r[3] if len(r) > 3 else -1
                clean_nodes.append((nid, x, y, cid))
                
            dt = np.dtype([('id', 'i4'), ('x', 'f8'), ('y', 'f8'), ('constraint_id', 'i4')])
            self.data['nodes'] = np.array(clean_nodes, dtype=dt)

        # Edges
        if self.data['edges']:
            # Assume: ID, N1, N2, BndID
            dt = np.dtype([('id', 'i4'), ('n1', 'i4'), ('n2', 'i4'), ('boundary_id', 'i4')])
            self.data['edges'] = np.array([tuple(r) for r in self.data['edges']], dtype=dt)

# --- 3. MAIN (The Driver) ---

def load_file(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    
    lexer = Lexer(source)
    parser = Parser(lexer)
    
    print("--- Starting Compilation ---")
    parser.program()
    print("--- Compilation Complete ---")
    
    return parser.data

if __name__ == "__main__":
    # Test with your file
    db = load_file("ex2.inp")
    
    print("\nRESULTING DATABASE:")
    for k, v in db.items():
        if isinstance(v, np.ndarray):
            print(f"[{k}] NumPy Array shape={v.shape}")
            print(v)
        else:
            print(f"[{k}] List/Dict: {v}")