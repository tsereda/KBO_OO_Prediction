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
        
        # Extract atom names from header
        atoms = []
        atom_pattern = r'\d+\s+([A-Z][a-z]*)'
        for match_obj in re.finditer(atom_pattern, lines[0]):
            atoms.append(match_obj.group(1))
        
        n_atoms = len(atoms)
        if n_atoms == 0:
            return {'matrix': [], 'atoms': []}
        
        matrix = np.zeros((n_atoms, n_atoms))
        
        # Parse distance matrix
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    row_idx = int(parts[0]) - 1
                    if row_idx < n_atoms:
                        clean_line = line.replace('*', '')
                        clean_parts = clean_line.split()
                        distances = clean_parts[2:]  # Skip index and atom name
                        
                        for col_idx, dist_str in enumerate(distances):
                            if col_idx < n_atoms:
                                try:
                                    dist = float(dist_str.strip())
                                    matrix[row_idx][col_idx] = dist
                                    matrix[col_idx][row_idx] = dist
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
    
    def extract_hybridization(self) -> Dict:
        """Extract orbital hybridization information."""
        pattern = r'PRINT OFF ORIENTATION INFORMATION FOR VALENCE ORBITAL CHARACTER PERCENT\.\s+(.*?)(?=\n\s*END OF CURRENT INFORMATION|$)'
        match = re.search(pattern, self.content, re.DOTALL)
        
        if not match:
            return {'orbitals': [], 'percent_s': [], 'percent_p': [], 'percent_d': [], 'percent_f': []}
        
        lines = match.group(1).strip().split('\n')
        if not lines:
            return {'orbitals': [], 'percent_s': [], 'percent_p': [], 'percent_d': [], 'percent_f': []}
        
        orbitals = []
        percent_s = []
        percent_p = []
        percent_d = []
        percent_f = []
        
        for line in lines:
            line = line.strip()
            if not line or 'ORB I' in line or 'PERCENT' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    orbital_idx = int(parts[0])
                    s_percent = float(parts[1])
                    p_percent = float(parts[2])
                    d_percent = float(parts[3])
                    f_percent = float(parts[4])
                    
                    orbitals.append(orbital_idx)
                    percent_s.append(s_percent)
                    percent_p.append(p_percent)
                    percent_d.append(d_percent)
                    percent_f.append(f_percent)
                    
                except (ValueError, IndexError):
                    continue
        
        return {
            'orbitals': orbitals,
            'percent_s': percent_s,
            'percent_p': percent_p,
            'percent_d': percent_d,
            'percent_f': percent_f
        }
    
    def extract_orbital_to_atom_mapping(self) -> Dict:
        """Extract orbital counts per atom. H atoms get 1 orbital, others get 4."""
        coords = self.extract_coordinates()
        atom_orbital_counts = []
        total_orbitals = 0
        
        # Try to extract from GAMESS output first
        pattern = r'FINAL ATOM NUMBERS:\s+(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)'
        match = re.search(pattern, self.content, re.DOTALL)
        
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                if 'ATOM =' in line and 'ACTIVE ORBITALS NUMBER=' in line:
                    parts = line.split()
                    orbital_count = int(parts[6])
                    atom_orbital_counts.append(orbital_count)
                    total_orbitals += orbital_count
        
        # Fill in missing atoms with standard counts
        n_atoms = len(coords['atoms'])
        while len(atom_orbital_counts) < n_atoms:
            atom_idx = len(atom_orbital_counts)
            atom = coords['atoms'][atom_idx]
            if atom == 'H':
                atom_orbital_counts.append(1)
                total_orbitals += 1
            else:
                atom_orbital_counts.append(4)
                total_orbitals += 4
        
        return {
            'atom_orbital_counts': atom_orbital_counts,
            'total_orbitals': total_orbitals
        }
    
    def extract_atomic_partial_charges(self) -> Dict:
        """Extract per-atom partial charges from orbital occupations."""
        density_matrix = self.extract_original_density_matrix()
        if density_matrix is None:
            return {'atoms': [], 'partial_charges': []}
        
        orbital_occupations = [density_matrix[i, i] for i in range(density_matrix.shape[0])]
        mapping = self.extract_orbital_to_atom_mapping()
        coords = self.extract_coordinates()
        
        if not mapping['atom_orbital_counts']:
            return {'atoms': [], 'partial_charges': []}
        
        partial_charges = []
        orbital_idx = 0
        
        for i, atom_type in enumerate(coords['atoms']):
            if i < len(mapping['atom_orbital_counts']):
                orbital_count = mapping['atom_orbital_counts'][i]
                
                if atom_type == 'H':
                    # Hydrogen: single orbital
                    if orbital_idx < len(orbital_occupations):
                        partial_charges.append(orbital_occupations[orbital_idx])
                    else:
                        partial_charges.append(0.0)
                else:
                    # Other atoms: sum all orbitals
                    atom_charge = 0.0
                    for j in range(orbital_count):
                        if orbital_idx + j < len(orbital_occupations):
                            atom_charge += orbital_occupations[orbital_idx + j]
                    partial_charges.append(atom_charge)
                
                orbital_idx += orbital_count
            else:
                partial_charges.append(0.0)
        
        return {
            'atoms': coords['atoms'],
            'partial_charges': partial_charges
        }
    
    def extract_kei_bo_orientation(self) -> Dict:
        """Extract KEI-BO orientation information."""
        pattern = r'PRINT OFF ORIENTATION INFORMATION BY KEI-BO MAGNITUDE\.\s+(.*?)(?=\n\s*END OF CURRENT INFORMATION|$)'
        match = re.search(pattern, self.content, re.DOTALL)
        
        if not match:
            return {'data': [], 'headers': []}
        
        lines = match.group(1).strip().split('\n')
        if not lines:
            return {'data': [], 'headers': []}
        
        # Find header line
        header_line = None
        data_start_idx = 0
        for i, line in enumerate(lines):
            if 'BOND ORDER' in line and 'KEI-BO' in line:
                header_line = line.strip()
                data_start_idx = i + 1
                break
        
        if header_line is None:
            return {'data': [], 'headers': []}
        
        headers = ['bond_order', 'kei_bo', 'orb_i', 'occ_i', 'atm_i', 'orbtyp_i', 'orb_j', 'occ_j', 'atm_j', 'orbtyp_j']
        data = []
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line or 'END OF CURRENT INFORMATION' in line:
                break
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            try:
                bond_order = float(parts[0])
                kei_bo = float(parts[1])
                orb_i = int(parts[2])
                occ_i = float(parts[3])
                
                remaining_line = line
                for _ in range(4):
                    remaining_line = remaining_line[remaining_line.find(' '):].lstrip()
                
                orb_j_match = re.search(r'\s+(\d+)\s+([+-]?\d*\.?\d+)', remaining_line)
                if not orb_j_match:
                    continue
                
                orb_j = int(orb_j_match.group(1))
                occ_j = float(orb_j_match.group(2))
                
                orbtyp_i_end = orb_j_match.start()
                orbtyp_i = remaining_line[:orbtyp_i_end].strip()
                
                orbtyp_j_start = orb_j_match.end()
                orbtyp_j = remaining_line[orbtyp_j_start:].strip()
                
                def parse_orbtyp(orbtyp_str):
                    match = re.match(r'([A-Z][a-z]*)\s+(\d+)\s+\(\s*([A-Z][a-z]*)\s+(\d+)\s*\)\s+(\w+)', orbtyp_str)
                    if match:
                        return {
                            'atom_type': match.group(1),
                            'atom_num': int(match.group(2)),
                            'parent_atom_type': match.group(3), 
                            'parent_atom_num': int(match.group(4)),
                            'orbital_type': match.group(5)
                        }
                    return orbtyp_str
                
                atm_i_info = parse_orbtyp(orbtyp_i)
                atm_j_info = parse_orbtyp(orbtyp_j)
                
                data.append({
                    'bond_order': bond_order,
                    'kei_bo': kei_bo,
                    'orb_i': orb_i,
                    'occ_i': occ_i,
                    'atm_i': atm_i_info,
                    'orbtyp_i': orbtyp_i,
                    'orb_j': orb_j,
                    'occ_j': occ_j,
                    'atm_j': atm_j_info,
                    'orbtyp_j': orbtyp_j
                })
                
            except (ValueError, IndexError):
                continue
        
        return {'data': data, 'headers': headers}

    def _parse_matrix(self, matrix_text: str) -> Optional[np.ndarray]:
        """Parse a matrix from GAMESS output format."""
        lines = [line.strip() for line in matrix_text.strip().split('\n') if line.strip()]
        if not lines:
            return None
        
        max_row = max((int(match.group(1)) for line in lines 
                      if (match := re.match(r'^\s*(\d+)\s+', line))), default=0)
        if max_row == 0:
            return None
        
        matrix = np.zeros((max_row, max_row))
        i = 0
        
        while i < len(lines):
            parts = lines[i].split()
            
            if parts and all(p.isdigit() for p in parts) and '.' not in lines[i]:
                col_indices = [int(x) - 1 for x in parts]
                i += 1
                
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
        kei_bo_data = self.extract_kei_bo_orientation()
        hybridization_data = self.extract_hybridization()
        partial_charges = self.extract_atomic_partial_charges()
        
        return {
            'coordinates': coords,
            'internuclear_distances': self.extract_internuclear_distances(),
            'original_density_matrix': density_matrix,
            'ke_density_matrix': ke_matrix,
            'kei_bo_orientation': kei_bo_data,
            'hybridization': hybridization_data,
            'atomic_partial_charges': partial_charges
        }
    
    def print_summary(self):
        """Print a formatted summary of the extracted data."""
        results = self.parse_all()
        
        self._print_coordinates(results['coordinates'])
        self._print_internuclear_distances(results['internuclear_distances'])
        self._print_hybridization(results['hybridization'])
        self._print_atomic_partial_charges(results['atomic_partial_charges'])
        self._print_matrix('Original Oriented Density Matrix', results['original_density_matrix'])
        self._print_matrix('Kinetic Energy Density Matrix', results['ke_density_matrix'])
        self._print_kei_bo_orientation(results['kei_bo_orientation'])
    
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
        
        print("      " + "".join(f"{atom:>10s}" for atom in atoms))
        
        for i, atom in enumerate(atoms):
            row = f"  {atom:4s}"
            for j in range(len(atoms)):
                if j <= i:
                    if i == j:
                        val = "---"
                    elif matrix[i][j] > 0:
                        val = f"{matrix[i][j]:10.6f}"
                    else:
                        val = "---"
                    row += f"{val:>10s}"
                else:
                    row += f"{'':>10s}"
            print(row)
    
    def _print_hybridization(self, hybrid_data: Dict):
        """Print orbital hybridization information."""
        if not hybrid_data or not hybrid_data['orbitals']:
            return
        
        print(f"\nOrbital Hybridization:")
        print()
        
        print("  " + f"{'Orbital':>8s} {'%S':>12s} {'%P':>12s}")
        print("  " + "-" * 35)
        
        for i in range(len(hybrid_data['orbitals'])):
            orbital = hybrid_data['orbitals'][i]
            s_pct = hybrid_data['percent_s'][i]
            p_pct = hybrid_data['percent_p'][i]
            
            print(f"  {orbital:8d} {s_pct:12.7f} {p_pct:12.7f}")
    
    
    def _print_atomic_partial_charges(self, charge_data: Dict):
        """Print atomic partial charges."""
        if not charge_data or not charge_data['atoms']:
            return
        
        print(f"\nAtomic Partial Charges ({len(charge_data['atoms'])} atoms):")
        print("  • H: Single s orbital occupation | Others: Sum of 4 valence orbital occupations")
        print()
        
        print("  " + f"{'Atom':>6s} {'Type':>6s} {'Partial Charge':>15s}")
        print("  " + "-" * 30)
        
        for i in range(len(charge_data['atoms'])):
            atom_type = charge_data['atoms'][i]
            partial_charge = charge_data['partial_charges'][i]
            print(f"  {i+1:6d} {atom_type:>6s} {partial_charge:15.7f}")
    
    def _print_matrix(self, name: str, matrix: Optional[np.ndarray]):
        """Print density matrix summary."""
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
                if j <= i:
                    row += f"{matrix[i][j]:10.6f}"
                else:
                    row += f"{'':>10s}"
            if matrix.shape[1] > 50:
                row += "  ..."
            print(row)
        
        if matrix.shape[0] > 50:
            print("  ...")
    
    def _print_kei_bo_orientation(self, kei_bo_data: Dict):
        """Print KEI-BO orientation information."""
        if not kei_bo_data or not kei_bo_data['data']:
            return
        
        print(f"\nKEI-BO Orientation Information ({len(kei_bo_data['data'])} entries):")
        print("  • Bond order and kinetic energy interaction analysis between orbitals")
        print("  • Positive/negative KEI-BO values indicate bonding/antibonding interactions")
        print()
        
        print("  " + f"{'Bond Order':>12s} {'KEI-BO':>12s} {'Orb I':>6s} {'Occ I':>10s} {'Orbital Type I':>20s} "
              f"{'Orb J':>6s} {'Occ J':>10s} {'Orbital Type J':>20s}")
        print("  " + "-" * 108)
        
        for entry in kei_bo_data['data']:
            if isinstance(entry['atm_i'], dict):
                atm_i_display = f"{entry['atm_i']['atom_type']}{entry['atm_i']['atom_num']}"
                orb_type_i = entry['atm_i']['orbital_type']
            else:
                atm_i_display = str(entry['atm_i'])
                orb_type_i = entry['orbtyp_i']
            
            if isinstance(entry['atm_j'], dict):
                atm_j_display = f"{entry['atm_j']['atom_type']}{entry['atm_j']['atom_num']}"
                orb_type_j = entry['atm_j']['orbital_type']
            else:
                atm_j_display = str(entry['atm_j'])
                orb_type_j = entry['orbtyp_j']
            
            orbital_i_display = f"{atm_i_display}({orb_type_i})"
            orbital_j_display = f"{atm_j_display}({orb_type_j})"
            
            print(f"  {entry['bond_order']:12.7f} {entry['kei_bo']:12.7f} "
                  f"{entry['orb_i']:6d} {entry['occ_i']:10.7f} {orbital_i_display:>20s} "
                  f"{entry['orb_j']:6d} {entry['occ_j']:10.7f} {orbital_j_display:>20s}")


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gamess_parser.py <filename>")
        sys.exit(1)
    
    parser = GAMESSParser(sys.argv[1])
    parser.print_summary()


if __name__ == "__main__":
    main()