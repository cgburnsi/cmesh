''' core/lexer.py
    -------------
    Lexical Analyzer (Tokenizer) for the SnapMesh Input Format.
    
    This module breaks the raw input text into a stream of tokens (Identifiers, 
    Numbers, Brackets) that the Parser can consume. It is designed to be 
    robust, simple, and dependency-free.
    

    Attribution & License:
    ----------------------
    This Lexer implementation is heavily inspired by and adapted from the 
    "Teeny Tiny Compiler" project by Austen Hensley.
    
    Source: https://github.com/AZHenley/teenytinycompiler (or specific repo URL)
    
    Copyright (c) 2020 Austen Hensley
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
'''
import sys
from enum import Enum, auto


class TokenType(Enum):
    """ Defines the valid token categories for the input file. """
    EOF = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    LBRACKET = auto()
    RBRACKET = auto()


class Token:
    """ A single unit of text (lexeme) with its classification. """
    def __init__(self, text, kind):
        self.text = text
        self.kind = kind


class Lexer:
    """
    Iterates through the input string and produces Tokens.
    
    Usage:
        >>> lexer = Lexer(" [nodes] 1 0.0 0.0 ")
        >>> token = lexer.getToken()
    """
    def __init__(self, input_text):
        self.source = input_text + '\n' # Append newline to simplify EOF handling
        self.curPos = -1
        self.curChar = ''
        self.nextChar()
    
    def nextChar(self):
        self.curPos += 1
        if self.curPos >= len(self.source):
            self.curChar = '\0'  # Null character indicates End of File
        else:
            self.curChar = self.source[self.curPos]

    def peek(self):
        if self.curPos + 1 >= len(self.source):
            return '\0'
        return self.source[self.curPos + 1]

    def abort(self, msg):
        sys.exit("Lexing error: " + msg)

    def skipWhitespace(self):
        while self.curChar in [' ', '\t', '\n', '\r']:
            self.nextChar()

    def skipComment(self):
        if self.curChar == '#':
            while self.curChar != '\n' and self.curChar != '\0':
                self.nextChar()

    def getToken(self):
        self.skipWhitespace()
        
        # Check for comments again in case whitespace led to one
        while self.curChar == '#':
            self.skipComment()
            self.skipWhitespace()

        if self.curChar == '\0':
            return Token('', TokenType.EOF)
        elif self.curChar == '[':
            self.nextChar()
            return Token('[', TokenType.LBRACKET)
        elif self.curChar == ']':
            self.nextChar()
            return Token(']', TokenType.RBRACKET)
        
        # Numbers (Integers, Floats, Negatives)
        elif self.curChar.isdigit() or self.curChar == '.' or self.curChar == '-':
            start = self.curPos
            # Peek ahead to capture the full number string
            while self.peek().isdigit() or self.peek() == '.' or self.peek() == '-':
                self.nextChar()
            text = self.source[start : self.curPos+1]
            self.nextChar()
            return Token(text, TokenType.NUMBER)
        
        # Identifiers (Section names, Enums)
        elif self.curChar.isalpha():
            start = self.curPos
            while self.peek().isalnum() or self.peek() == '_':
                self.nextChar()
            text = self.source[start : self.curPos+1]
            self.nextChar()
            return Token(text, TokenType.IDENTIFIER)
        
        else:
            self.abort(f"Unknown token: {self.curChar}")

if __name__ == '__main__':
    # Simple test driver
    sample = "[test] 123 45.6 # comment"
    lexer = Lexer(sample)
    t = lexer.getToken()
    while t.kind != TokenType.EOF:
        print(f"{t.kind}: {t.text}")
        t = lexer.getToken()