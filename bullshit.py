# Run this once to fix the file on disk
clean_geometry = """[nodes]
1   0.0   0.0
2   0.0   2.0
3   1.0   2.0
4   2.0   1.0
5   3.0   2.0
6   3.0   0.0

[constraints]
1      line     0.0     0.0     0.0

[faces]
1   1   2   1
2   2   3   1
3   3   4   1
4   4   5   1
5   5   6   1
6   6   1   1

[cells]
"""

with open('geom1.inp', 'w') as f:
    f.write(clean_geometry)

print("geom1.inp has been overwritten with clean data.")