import os
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from ecoli_db_builder import build_sequence_group_by_length

def create_fast_file(seq_list: List[str], fasta_save_path: str):
    """
    Each item in seq_list is a sequence string.
    Creates a FASTA file from the given list of sequences
    and saves it to fasta_save_path.
    """
    # Create a list of SeqRecord objects
    records = []
    for i, seq in enumerate(seq_list, start=1):
        record = SeqRecord(
            Seq(seq),
            id=f"seq_{i}",       # unique sequence ID
            description=""       # optional description
        )
        records.append(record)
    
    # Write the sequences to a FASTA file
    with open(fasta_save_path, "w") as fasta_file:
        SeqIO.write(records, fasta_file, "fasta")
    
    print(f"FASTA file saved with {len(records)} sequences at: {fasta_save_path}")

def build_fasta_db(csv_path, fasta_save_path):
    '''
    Using CSV Path it creates Fasta files grouped by sequence lenght
    and saves.  
    '''
    seq_groups = build_sequence_group_by_length(csv_path)

    os.makedirs(fasta_save_path, exist_ok=True)

    for group_name, seq_list in seq_groups.items():
        file_name = f"ecoli_{group_name}.fasta"
        save_full_path = os.path.join(fasta_save_path, file_name)
        create_fast_file(seq_list, save_full_path)

def create_fasta_file_from_single_seq(sequence, save_full_path, seq_name="sequence"):
    """
    Creates a FASTA file containing a single sequence.

    Args:
        sequence (str): Amino acid or nucleotide sequence.
        save_full_path (str): Full path including filename where FASTA will be saved.
    """
    # Ensure sequence is a single string without spaces/newlines
    sequence = sequence.replace(" ", "").replace("\n", "")
    
    # Open file and write FASTA
    with open(save_full_path, "w") as f:
        f.write(f">{seq_name}\n")
        
        # Optional: wrap sequence every 60 characters (standard FASTA format)
        for i in range(0, len(sequence), 60):
            f.write(sequence[i:i+60] + "\n")


def build_fasta_per_sequence(csv_path, fasta_save_path):
    '''
    Generates one fasta file per sequence for each group.
    and saves.  
    '''
    seq_groups = build_sequence_group_by_length(csv_path)

    for group_name, seq_list in seq_groups.items():
        #Create folder 
        group_path = f"{fasta_save_path}/{group_name}"
        os.makedirs(group_path, exist_ok=True)

        for index, sequence in enumerate(seq_list, start=1):
            file_name = f"ecoli_{group_name}_{index}.fasta"

            save_full_path = os.path.join(fasta_save_path, group_name, file_name)
            create_fasta_file_from_single_seq(sequence, save_full_path, seq_name=f"seq_{index}")
    


if __name__ == "__main__":
    csv_path = 'data/ecoli_normalized.csv'
    fasta_save_path = "data/fasta"
    # build_fasta_db(csv_path, fasta_save_path)
    build_fasta_per_sequence(csv_path, fasta_save_path)
    
