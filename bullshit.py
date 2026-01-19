import os
import sys
from core.input_reader import Lexer, Parser, TokenType

filename = 'geom1.inp'

print(f"--- DIAGNOSTIC: Checking {filename} ---")

# 1. Check File Existence and Content
if not os.path.exists(filename):
    print(f"ERROR: {filename} does not exist in {os.getcwd()}")
    sys.exit(1)

with open(filename, 'r') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")
print(f"First 50 chars: {repr(content[:50])}")

if len(content.strip()) == 0:
    print("ERROR: File is empty or whitespace only.")
    sys.exit(1)

# 2. Check Lexer Output (First 10 Tokens)
print("\n--- DIAGNOSTIC: Lexer Output ---")
lexer = Lexer(content)
for i in range(10):
    token = lexer.getToken()
    print(f"Token {i}: {token.kind} ('{token.text}')")
    if token.kind == TokenType.EOF:
        break

# 3. Check Parser Output
print("\n--- DIAGNOSTIC: Parser Output ---")
# Reset lexer for parser
lexer = Lexer(content)
parser = Parser(lexer)

try:
    parser.program()
    print(f"Nodes Found: {len(parser.nodes)}")
    print(f"Faces Found: {len(parser.faces)}")
    print(f"Constraints Found: {len(parser.constraints)}")
    print(f"Sources Found: {len(parser.sources)}")
except Exception as e:
    print(f"PARSER CRASHED: {e}")