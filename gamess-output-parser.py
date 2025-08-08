import re
import numpy as np
from typing import Dict, List, Optional, Tuple

class GAMESSParser:
    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, 'r') as f:
            self.content = f.read()
    
    def extract_coordinates(self) -> Dict:
        """Extract atomic coordinates in both Bohr and Angstrom units."""
        coords = {'bohr': [], 'atoms': [], 'atomic_charges': []}
        
        pattern = r'ATOM\s+ATOMIC\s+COORDINATES \(BOHR\)\s+CHARGE\s+X\s+Y\s+Z\s+(.*?)(?=\n\s*\n|\n\s*INTERNUCLEAR|$)'
        match = re.search(pattern, self.content, re.DOTALL)
        
        if match:
            for line in match.group(1).strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        coords['atoms'].append(parts[0])
                        coords['atomic_charges'].append(float(parts[1]))
                        coords['bohr'].append([float(x) for x in parts[2:5]])
        
        # Convert to Angstrom
        bohr_to_ang = 0.529177
        coords['angstrom'] = [[x*bohr_to_ang for x in coord] for coord in coords['bohr']]
        return coords
    
    def extract_internuclear_distances(self) -> Dict:
        """Extract internuclear distances matrix."""
        pattern = r'INTERNUCLEAR DISTANCES \(ANGS\.\)\s+[-]+\s+(.*?)(?=\n\s*\*|\n\s*\n\s*\*|$)'
        match = re.search(pattern, self.content, re.DOTALL)
        
        if not match:
            return {'matrix': [], 'atoms': []}
        
        lines = match.group(1).strip().split('\n')
        if not lines:
            return {'matrix': [], 'atoms': []}
        
        # Extract atom names from the header line
        atoms = []
        atom_pattern = r'\d+\s+([A-Z][a-z]*)'
        for match_obj in re.finditer(atom_pattern, lines[0]):
            atoms.append(match_obj.group(1))
        
        n_atoms = len(atoms)
        if n_atoms == 0:
            return {'matrix': [], 'atoms': []}
        
        matrix = np.zeros((n_atoms, n_atoms))
        
        # Parse the distance matrix
        for line in lines[1:]:  # Skip header line
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    row_idx = int(parts[0]) - 1  # Convert to 0-based index
                    if row_idx < n_atoms:
                        # Skip the atom name (parts[1]) and parse distances
                        # Remove asterisks from the entire line first
                        clean_line = line.replace('*', '')
                        clean_parts = clean_line.split()
                        
                        # Extract just the distance values (skip row index and atom name)
                        distances = clean_parts[2:]  # Skip index and atom name
                        
                        for col_idx, dist_str in enumerate(distances):
                            if col_idx < n_atoms:
                                try:
                                    dist = float(dist_str.strip())
                                    matrix[row_idx][col_idx] = dist
                                    matrix[col_idx][row_idx] = dist  # Symmetric matrix
                                except ValueError:
                                    continue
                except (ValueError, IndexError):
                    continue
        
        return {'matrix': matrix.tolist(), 'atoms': atoms}
    
    def _extract_matrix(self, pattern: str) -> Optional[np.ndarray]:
        """Extract matrix using given regex pattern."""
        match = re.search(pattern, self.content, re.DOTALL)
        return self._parse_matrix(match.group(1)) if match else None
    
    def extract_original_density_matrix(self) -> Optional[np.ndarray]:
        """Extract the original oriented density matrix."""
        pattern = r'ORIGINAL ORIENTED DENSITY MATRIX\s+(.*?)(?=\n\s*MULTIPLY|\n\s*\n\s*[A-Z]|$)'
        return self._extract_matrix(pattern)
    
    def extract_ke_density_matrix(self) -> Optional[np.ndarray]:
        """Extract the kinetic energy density matrix."""
        pattern = r'FULL PRINT OUT OF NOVEL ORIENTED DENSITY\s+(.*?)(?=\n\s*OLD TOTAL|\n\s*\n\s*[A-Z]|$)'
        return self._extract_matrix(pattern)

    def _parse_matrix(self, matrix_text: str) -> Optional[np.ndarray]:
        """Parse a matrix from GAMESS output format."""
        lines = [line.strip() for line in matrix_text.strip().split('\n') if line.strip()]
        if not lines:
            return None
        
        # Find matrix size
        max_row = max((int(match.group(1)) for line in lines 
                      if (match := re.match(r'^\s*(\d+)\s+', line))), default=0)
        if max_row == 0:
            return None
        
        matrix = np.zeros((max_row, max_row))
        i = 0
        
        while i < len(lines):
            parts = lines[i].split()
            
            # Check for column header (all digits, no decimals)
            if parts and all(p.isdigit() for p in parts) and '.' not in lines[i]:
                col_indices = [int(x) - 1 for x in parts]
                i += 1
                
                # Parse data rows for this block
                while i < len(lines):
                    row_match = re.match(r'^\s*(\d+)\s+(.*)', lines[i])
                    if row_match:
                        row_idx = int(row_match.group(1)) - 1
                        values = [float(x) for x in row_match.group(2).split()]
                        
                        for j, val in enumerate(values):
                            if j < len(col_indices):
                                col_idx = col_indices[j]
                                matrix[row_idx][col_idx] = matrix[col_idx][row_idx] = val
                        i += 1
                    else:
                        break
            else:
                i += 1
        
        return matrix
    
    def parse_all(self) -> Dict:
        """Parse all information from the GAMESS output file."""
        coords = self.extract_coordinates()
        density_matrix = self.extract_original_density_matrix()
        ke_matrix = self.extract_ke_density_matrix()
        
        return {
            'coordinates': coords,
            'internuclear_distances': self.extract_internuclear_distances(),
            'original_density_matrix': density_matrix,
            'ke_density_matrix': ke_matrix
        }
    
    def print_summary(self):
        """Print a formatted summary of the extracted data."""
        results = self.parse_all()
        
        # Coordinates
        self._print_coordinates(results['coordinates'])
        
        # Internuclear distances
        self._print_internuclear_distances(results['internuclear_distances'])
        
        # Density matrices (extracted directly from GAMESS output)
        self._print_matrix('Original Oriented Density Matrix', results['original_density_matrix'])
        self._print_matrix('Kinetic Energy Density Matrix', results['ke_density_matrix'])
    
    def _print_coordinates(self, coords):
        """Print molecular geometry."""
        if not coords['atoms']:
            return
        
        print(f"Molecular Geometry ({len(coords['atoms'])} atoms):")
        print("  Atom    Charge      X(Bohr)      Y(Bohr)      Z(Bohr)      X(Å)         Y(Å)         Z(Å)")
        print("  " + "-"*88)
        
        for i, atom in enumerate(coords['atoms']):
            bohr, ang = coords['bohr'][i], coords['angstrom'][i]
            charge = coords['atomic_charges'][i]
            print(f"  {atom:4s}    {charge:5.1f}    {bohr[0]:10.6f}   {bohr[1]:10.6f}   {bohr[2]:10.6f}   "
                  f"{ang[0]:10.6f}   {ang[1]:10.6f}   {ang[2]:10.6f}")
    
    def _print_internuclear_distances(self, distances):
        """Print internuclear distances matrix."""
        if not distances['matrix'] or not distances['atoms']:
            return
        
        print(f"\nInternuclear Distances (Å):")
        matrix, atoms = np.array(distances['matrix']), distances['atoms']
        
        # Header
        print("      " + "".join(f"{atom:>10s}" for atom in atoms))
        
        # Matrix rows (lower triangle only)
        for i, atom in enumerate(atoms):
            row = f"  {atom:4s}"
            for j in range(len(atoms)):
                if j <= i:  # Only print lower triangle (including diagonal)
                    if i == j:
                        val = "---"  # Diagonal elements
                    elif matrix[i][j] > 0:
                        val = f"{matrix[i][j]:10.6f}"
                    else:
                        val = "---"
                    row += f"{val:>10s}"
                else:
                    row += f"{'':>10s}"  # Empty space for upper triangle
            print(row)
    
    def _print_matrix(self, name: str, matrix: Optional[np.ndarray]):
        """Print density matrix summary with explanations."""
        if matrix is None:
            return
        
        print(f"\n{name} ({matrix.shape[0]}x{matrix.shape[1]}):")
        
        if "Original Oriented Density" in name:
            print("  • Diagonal: Electron populations | Off-diagonal: Orbital overlap/mixing")
            print(f"  • Total electrons: {np.trace(matrix):.6f}")
            
        elif "Kinetic Energy Density" in name:
            print("  • Diagonal: Kinetic energy contributions | Off-diagonal: KEI-BO values")
            print(f"  • Total kinetic energy: {np.trace(matrix):.6f} a.u.")
        
        print()
        max_display = min(matrix.shape[0], 50)
        
        for i in range(max_display):
            row = f"  Row {i+1:2d}:"
            for j in range(min(matrix.shape[1], 50)):
                if j <= i:  # Only print lower triangle (including diagonal)
                    row += f"{matrix[i][j]:10.6f}"
                else:
                    row += f"{'':>10s}"  # Empty space for upper triangle
            if matrix.shape[1] > 50:
                row += "  ..."
            print(row)
        
        if matrix.shape[0] > 50:
            print("  ...")


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gamess_parser.py <filename>")
        sys.exit(1)
    
    parser = GAMESSParser(sys.argv[1])
    parser.print_summary()


if __name__ == "__main__":
    main()