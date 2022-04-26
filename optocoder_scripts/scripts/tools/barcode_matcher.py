import pandas as pd
import numpy as np
def get_cycle_scores(optical_path, num_cycles):
        optical = pd.read_csv(optical_path, sep="\t")

        optical_scores = optical.iloc[:, 4:]
        optical_scores = np.array(optical_scores)
        optical_scores = np.hsplit(optical_scores, num_cycles)

        cycle_scores = []
        for cycle in optical_scores:
                maxes = np.max(cycle, axis=1)
                cycle_scores.append(maxes)

        cycle_scores = np.array(cycle_scores)
        return cycle_scores

def map_back(barcode):
    nucs = list(barcode)
    mapper = ['T', 'G', 'C', 'A', 'N']
    new_barcode = []
    for nuc in nucs:
        new_barcode.append(str(mapper.index(nuc)))

    return ''.join(new_barcode)

def convert_to_color_space(nuc, next_nuc):
    if nuc == 'A':
        return str(['A', 'C', 'G', 'T', 'N'].index(next_nuc))
    elif nuc == 'C':
        return str(['C', 'A', 'T', 'G', 'N'].index(next_nuc))
    elif nuc == 'G':
        return str(['G', 'T', 'A', 'C', 'N'].index(next_nuc))
    elif nuc == 'T':
        return str(['T', 'G', 'C', 'A', 'N'].index(next_nuc))
    elif nuc == 'N':
        return str(4)

def convert_barcodes(barcode, ligation_seq):
    nucleotides = list(barcode)
    part_1 = nucleotides[:6]
    bc_1 = ['T']
    bc_1.extend(part_1)
    bc_1.extend('T')

    part_2 = nucleotides[6:]
    bc_2 = ['A']
    bc_2.extend(part_2)
    bc_2.extend('')

    bc_1_cs = []
    bc_2_cs = []
    for i in range(len(bc_1) - 1):
        bc_1_cs.extend(convert_to_color_space(bc_1[i], bc_1[i+1]))

    for i in range(len(bc_2) - 1):
        bc_2_cs.extend(convert_to_color_space(bc_2[i], bc_2[i+1]))

    bc_1 = list(np.array(bc_1_cs)[ligation_seq])
    bc_2 = list(np.array(bc_2_cs)[ligation_seq])
    bc_1.extend(bc_2)
    return bc_1

class BarcodeMatcher():

    def match(self, optical_bc_path, illumina_path, is_solid, lig_seq=None, nuc_seq=None, is_optocoder=True, num_illumina_to_use=np.array([]), optical_loaded=False):
        # read illumina barcodes
        illumina_barcodes = np.genfromtxt(illumina_path, dtype='str')

        # read optical barcodes
        if is_optocoder:
            if optical_loaded:
                optical_barcodes = optical_bc_path['barcodes']
            else:
                optical_barcodes = pd.read_csv(optical_bc_path, sep='\t')['barcodes']
        else:
            if is_solid:
                allbarcodes = pd.read_csv(optical_bc_path, header=None)
                mapper = {'B':'T', 'G':'G', 'O':'C', 'R':'A', 'N':'N'}
                allbarcodes = allbarcodes.applymap(lambda x: mapper[x])
                allbarcodes['barcodes'] = allbarcodes.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
                optical_barcodes = list(allbarcodes['barcodes'])
                lig_seq = np.array([0,1,2,3,4,5,6])
                nuc_seq = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            else:   
                optical_barcodes = pd.read_csv(optical_bc_path, skiprows=1)
                optical_barcodes = list(optical_barcodes.iloc[:,0])

        if is_solid:
            optical_barcodes, illumina_barcodes = self._illumina_to_colorspace(optical_barcodes, illumina_barcodes, lig_seq, nuc_seq)

        if len(num_illumina_to_use) > 0:
            match_counts = []
            for num_ill in num_illumina_to_use:      
                illumina_barcodes_subs = illumina_barcodes[:num_ill]
                num_matches = self._match(optical_barcodes, illumina_barcodes_subs)
                match_counts.append(num_matches)
            return match_counts
        else:
            return self._match(optical_barcodes, illumina_barcodes)

    def _illumina_to_colorspace(self, optical_barcodes, illumina_barcodes, lig_seq, nuc_seq):
        colorspace_optical = [map_back(barcode) for barcode in optical_barcodes]

        illumina = []
        for barcode in illumina_barcodes:
            color_barcode = convert_barcodes(barcode, lig_seq)
            illumina.append(''.join(color_barcode))

        filtered_barcodes = []
        for barcode in colorspace_optical:
            nucs = np.array(list(barcode))
            nucs = nucs[nuc_seq]
            filtered_barcodes.append(''.join(nucs))
        barcodes = filtered_barcodes
        illumina_barcodes = illumina
        return barcodes, illumina_barcodes

    def _match(self, optical_barcodes, illumina_barcodes):
        return len(set(optical_barcodes).intersection(illumina_barcodes))

    def get_scores(self, optical_bc_path, illumina_path, bc_type, puck_name):
        optical = pd.read_csv(optical_bc_path, sep='\t')
        illumina = np.genfromtxt(illumina_path, dtype='str')
        op_bcs = optical['barcodes']
        matches = set(op_bcs).intersection(illumina)
        indices = optical[optical['barcodes'].isin(matches)]['bead_id']
        puck_scores =  get_cycle_scores(optical_bc_path, len(op_bcs[0]))
        mean_scores = np.mean(puck_scores, axis=0)  
        frame = pd.DataFrame({'Basecalling Type': np.repeat(bc_type, mean_scores.shape[0]), 'Puck': np.repeat(puck_name, mean_scores.shape[0]
), 'Score': mean_scores.flatten(), 'matching': np.repeat('Non-Matching', mean_scores.shape[0])})
        frame['matching'][indices] = 'Match'
        return frame