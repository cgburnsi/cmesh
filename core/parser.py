''' core/parser.py
    --------------
    Recursive Descent Parser for the SnapMesh Input Format.
    
    This module consumes the stream of tokens from the Lexer and constructs
    the structured NumPy arrays (Nodes, Faces, Fields) required by the core 
    engine. It uses a "Dispatch Table" pattern to cleanly separate the logic 
    for each section of the input file.
'''
import sys
import numpy as np
from .data_types import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE, CONSTRAINT_DTYPE, BC_DTYPE, FIELD_DTYPE
from .lexer import Lexer, TokenType 

class Parser:
    """
    Parses the input file into structured data arrays.
    
    Architecture:
    - Recursive Descent: The 'program' method calls 'process_section', which 
      dispatches to specific handlers (e.g., 'parse_nodes').
    - Dispatch Table: Section handlers are registered in 'self.section_handlers',
      making it easy to add new features without breaking the core loop.
    """
    def __init__(self, lexer):
        self.lexer = lexer
        self.curToken = None; self.peekToken = None
        self.nextToken(); self.nextToken()
        
        # 1. Data Storage
        self.data = {
            "settings": {"mode": "axisymmetric"}, # Default setting
            "fluids": {},        # NEW: Storage for fluid properties
            "nodes": [], 
            "faces": [], 
            "boundaries": [],
            "constraints": [], 
            "fields": []
        }
        
        # 2. Lookup Tables
        self.constraint_types = {'fixed':0, 'line':1, 'circle':2}
        self.field_types = {'global':0, 'box':1}
        self.bc_types = {'dirichlet':1, 'neumann':2}

        # 3. Dispatch Table
        self.section_handlers = {
            "settings": self.parse_settings,
            "fluids": self.parse_fluids,   # NEW: Handler for [fluids]
            "nodes": self.parse_nodes,
            "faces": self.parse_faces,
            "boundaries": self.parse_boundaries,
            "constraints": self.parse_constraints,
            "fields": self.parse_fields,
            "cells": self.skip_section 
        }
        
    # --- Core Parsing Helpers ---
    def checkToken(self, kind): return self.kind == kind
    
    def nextToken(self):
        self.curToken = self.peekToken; self.peekToken = self.lexer.getToken()
        self.kind = self.curToken.kind if self.curToken else None
        
    def match(self, kind):
        if not self.checkToken(kind): 
            got = self.curToken.text if self.curToken else 'None'
            self.abort(f"Expected {kind}, got {got}")
        self.nextToken()
        
    def abort(self, msg): 
        sys.exit(msg)

    # --- Value Extraction Helpers ---
    def expect_int(self):
        val = int(self.curToken.text); self.match(TokenType.NUMBER)
        return val
        
    def expect_float(self):
        val = float(self.curToken.text); self.match(TokenType.NUMBER)
        return val
    
    def expect_choice(self, options, default=None):
        if self.checkToken(TokenType.IDENTIFIER):
            val = self.curToken.text.lower()
            if val in options:
                self.match(TokenType.IDENTIFIER)
                return val
            else:
                self.abort(f"Invalid choice: '{val}'. Expected one of {options}")
        
        if default is not None: return default
        self.abort(f"Expected an identifier choice from {options}")

    def expect_enum(self, mapping_dict, default_val):
        val = default_val
        if self.checkToken(TokenType.IDENTIFIER):
            text = self.curToken.text.lower()
            val = mapping_dict.get(text, default_val)
            self.match(TokenType.IDENTIFIER)
        elif self.checkToken(TokenType.NUMBER):
            val = int(self.curToken.text); self.match(TokenType.NUMBER)
        return val

    # --- Main Loop ---
    def program(self):
        while not self.checkToken(TokenType.EOF): 
            self.process_section()

    def process_section(self):
        self.match(TokenType.LBRACKET)
        section_name = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.RBRACKET)

        handler = self.section_handlers.get(section_name)
        if not handler:
            self.abort(f"Unknown section: [{section_name}]")
        
        while self.checkToken(TokenType.NUMBER) or self.checkToken(TokenType.IDENTIFIER):
            handler()

    # --- Individual Handlers ---
    def parse_settings(self):
        key = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        
        if key == "mode":
            val = self.expect_choice(['planar', 'axisymmetric'], default='axisymmetric')
            self.data['settings'][key] = val
        else:
            val = self.curToken.text.lower()
            self.match(TokenType.IDENTIFIER)
            self.data['settings'][key] = val
        
    def parse_fluids(self):
        """ Parses fluid name, gas constant (R), and gamma. """
        fluid_name = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        r_gas = self.expect_float()
        gamma = self.expect_float()
        self.data['fluids'][fluid_name] = {"R": r_gas, "gamma": gamma}
        
    def parse_nodes(self):
        nid = self.expect_int(); x = self.expect_float(); y = self.expect_float()
        self.data['nodes'].append((nid, x, y))


    def parse_faces(self):
        """ Parses: ID n1 n2 BC_Tag C_Tag Segments """
        fid  = self.expect_int()
        n1, n2 = self.expect_int(), self.expect_int()
        tag  = self.expect_int()
        ctag = self.expect_int() # Read the new Constraint Tag
        segs = self.expect_int() 
        self.data['faces'].append((fid, n1, n2, tag, ctag, segs))



    def parse_boundaries(self):
        """ 
        Parses boundary primitives (u, v, p, T). 
        Density (rho) is set to 0.0 as a placeholder to match BC_DTYPE.
        """
        bid   = self.expect_int()
        b_raw = self.expect_choice(['dirichlet', 'neumann'])
        btype = self.bc_types.get(b_raw, 1)
        
        # Sequentially extract the 4 primitives required by the updated format
        u   = self.expect_float() # Axial Velocity
        v   = self.expect_float() # Radial Velocity
        p   = self.expect_float() # Pressure
        T   = self.expect_float() # Temperature
                
        # Append 7-element tuple matching BC_DTYPE: (id, type, rho, u, v, p, T)
        self.data['boundaries'].append((bid, btype, 0.0, u, v, p, T))



    def parse_constraints(self):
        """ Parses a geometry entity with up to 5 parameters """
        cid   = self.expect_int()
        ctype_str = self.expect_choice(['fixed', 'line', 'circle', 'arc'])
        ctype = self.constraint_types.get(ctype_str, 1)
        
        # Read the rest of the line as floats until the next ID or Section
        params = []
        while self.checkToken(TokenType.NUMBER):
            params.append(self.expect_float())
        
        # Pad with zeros to fit the 5-parameter DTYPE
        while len(params) < 5:
            params.append(0.0)
            
        self.data['constraints'].append((cid, ctype, *params[:5]))



    def parse_fields(self):
        sid = self.expect_int(); stype = self.expect_enum(self.field_types, 0)
        x1 = self.expect_float(); y1 = self.expect_float()
        x2 = self.expect_float(); y2 = self.expect_float(); v = self.expect_float()
        self.data['fields'].append((sid, stype, x1, y1, x2, y2, v))

    def skip_section(self):
        self.nextToken() 

    def get_arrays(self):
        """ Returns the data as structured NumPy arrays using the updated dtypes. """
        return {
            "settings":    self.data['settings'],
            "fluids":      self.data['fluids'],     # Return the fluids dict
            "nodes":       np.array(self.data['nodes'], dtype=NODE_DTYPE),
            "faces":       np.array(self.data['faces'], dtype=FACE_DTYPE),
            "boundaries":  np.array(self.data['boundaries'], dtype=BC_DTYPE),
            "constraints": np.array(self.data['constraints'], dtype=CONSTRAINT_DTYPE),
            "fields":      np.array(self.data['fields'], dtype=FIELD_DTYPE)
        }

def input_reader(filename):
    with open(filename, 'r') as f: 
        lexer  = Lexer(f.read())
        parser = Parser(lexer)
        parser.program()
        return parser.get_arrays()