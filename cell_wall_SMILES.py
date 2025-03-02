#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse
import time
from collections import Counter
import re

class CellWallSMILESPredictor:
    def __init__(self, output_dir='./smiles_output'):
        """Initialize the cell wall predictor that outputs a single SMILES"""
        # Create output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Representative SMILES for different cell wall types
        # These are simplified representations of the core structures
        self.cell_wall_smiles = {
            'gram_positive': "CC(=O)NC1C(OC2C(CO)OC(OC3C(O)C(O)C(OC(C)=O)C(OC(=O)CO)O3)C2OC(C)=O)OC(CO)C(O)C1O", # Peptidoglycan with teichoic acid
            'gram_negative': "CC(=O)NC1C(OC2C(CO)OC(OP(=O)(O)OCC(O)C(O)C(O)C(O)CO)C2O)OC(CO)C(O)C1NC(C)=O", # Peptidoglycan with LPS core
            'mycobacteria': "CCCCCCCCCCCCCCCC(=O)OC1CC(OC2OC(CO)C(O)C(O)C2O)OC(CO)C1O" # Mycolic acid-arabinogalactan
        }
        
        # Dictionary of genes associated with different cell wall types
        self.cell_wall_genes = {
            'gram_positive': [
                'murA', 'murB', 'murC', 'murD', 'murE', 'murF',  # Peptidoglycan synthesis
                'pbp1', 'pbp2', 'pbp3', 'pbp4',                   # Penicillin-binding proteins
                'tag', 'tarO', 'tarA', 'tarB',                    # Teichoic acid synthesis
                'dlt', 'dltA', 'dltB', 'dltC', 'dltD',            # D-alanylation
                'femX', 'femA', 'femB',                           # Additional Gram+ markers
                'graR', 'graS', 'vraF', 'vraG'                    # Cell wall stress response
            ],
            'gram_negative': [
                'lpxA', 'lpxB', 'lpxC', 'lpxD',                  # Lipid A biosynthesis
                'waa', 'waaA', 'waaC', 'waaF', 'waaG', 'waaP',   # Core oligosaccharide synthesis
                'rfa', 'rfaD', 'rfaF', 'rfaG', 'rfaP',           # LPS inner core
                'bam', 'bamA', 'bamB', 'bamC', 'bamD', 'bamE',   # Outer membrane protein assembly
                'tol', 'tolA', 'tolB', 'tolC', 'tolQ', 'tolR',   # Outer membrane integrity
                'omp', 'ompA', 'ompC', 'ompF', 'ompT',           # Outer membrane porins
                'lpt', 'lptA', 'lptB', 'lptC', 'lptD'            # LPS transport
            ],
            'mycobacteria': [
                'pks', 'pks13', 'fadD32', 'accD4', 'accD5',      # Mycolic acid synthesis
                'emb', 'embA', 'embB', 'embC', 'embR',           # Arabinogalactan synthesis
                'mmp', 'mmpL3', 'mmpL7', 'mmpL11',               # Mycolic acid transport
                'fbp', 'fbpA', 'fbpB', 'fbpC',                   # Antigen 85 complex
                'kas', 'kasA', 'kasB', 'inhA'                    # Fatty acid synthesis
            ]
        }
        
        # Compile regex patterns for faster matching
        self.gene_patterns = {}
        for cell_type, genes in self.cell_wall_genes.items():
            for gene in genes:
                # Create case-insensitive pattern
                self.gene_patterns[gene] = re.compile(gene, re.IGNORECASE)
        
    def process_fastq(self, fastq_file, max_reads=100000):
        """
        Process FASTQ file and identify cell wall genes
        Returns the counts of genes by cell wall type
        """
        print(f"Processing FASTQ file: {fastq_file}")
        start_time = time.time()
        
        # Dictionary to store gene counts by cell wall type
        type_counts = {
            'gram_positive': 0,
            'gram_negative': 0,
            'mycobacteria': 0
        }
        
        # Dictionary to track which specific genes were found
        genes_found = {
            'gram_positive': set(),
            'gram_negative': set(),
            'mycobacteria': set()
        }
        
        # Process FASTQ file with progress updates
        try:
            total_reads = 0
            progress_interval = 10000  # Update progress every 10,000 reads
            
            for i, record in enumerate(SeqIO.parse(fastq_file, "fastq")):
                if i >= max_reads:
                    print(f"Reached maximum reads limit ({max_reads})")
                    break
                    
                if i % progress_interval == 0 and i > 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {i:,} reads ({elapsed:.1f} seconds)")
                
                total_reads += 1
                
                # Get sequence
                sequence = str(record.seq).upper()
                
                # Check for cell wall gene patterns
                for gene, pattern in self.gene_patterns.items():
                    if pattern.search(sequence):
                        # Determine which cell wall type this gene belongs to
                        for cell_type, genes in self.cell_wall_genes.items():
                            if gene in genes:
                                type_counts[cell_type] += 1
                                genes_found[cell_type].add(gene)
                                break
            
            # Final progress update
            elapsed = time.time() - start_time
            print(f"Finished processing {total_reads:,} reads in {elapsed:.1f}s")
            
            # If no reads found, return empty results
            if total_reads == 0:
                print(f"Warning: No reads found in {fastq_file}")
                return None, None, None
            
            # Calculate percentages
            type_percentages = {}
            unique_genes_found = {}
            
            for cell_type in type_counts.keys():
                if len(self.cell_wall_genes[cell_type]) > 0:
                    unique_genes_found[cell_type] = len(genes_found[cell_type])
                    type_percentages[cell_type] = len(genes_found[cell_type]) / len(self.cell_wall_genes[cell_type]) * 100
                else:
                    unique_genes_found[cell_type] = 0
                    type_percentages[cell_type] = 0
            
            return type_counts, type_percentages, unique_genes_found
            
        except Exception as e:
            print(f"Error processing FASTQ file: {e}")
            return None, None, None
    
    def predict_cell_wall_type(self, fastq_file, max_reads=100000):
        """
        Predict cell wall type and return a single SMILES representation
        """
        print(f"Predicting cell wall type for: {fastq_file}")
        
        # Process FASTQ
        type_counts, type_percentages, unique_genes_found = self.process_fastq(fastq_file, max_reads)
        
        if type_counts is None:
            print(f"Error processing {fastq_file}")
            return None
            
        # Determine most likely cell wall type based on unique genes percentage
        if sum(type_percentages.values()) == 0:
            print("No cell wall genes found. Unable to predict cell wall type.")
            predicted_type = None
            confidence = 0
        else:
            predicted_type = max(type_percentages, key=type_percentages.get)
            
            # Calculate confidence (0-1)
            total_percentage = sum(type_percentages.values())
            confidence = type_percentages[predicted_type] / total_percentage if total_percentage > 0 else 0
        
        # Generate results
        result = {
            'filename': fastq_file,
            'predicted_type': predicted_type,
            'confidence': confidence,
            'counts': type_counts,
            'percentages': type_percentages,
            'unique_genes': unique_genes_found,
            'smiles': self.cell_wall_smiles.get(predicted_type, "")
        }
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(fastq_file))[0]
        
        # Save SMILES to file
        smiles_file = os.path.join(self.output_dir, f"{base_name}_cell_wall.smi")
        with open(smiles_file, 'w') as f:
            if predicted_type:
                # Standard SMILES format: SMILES string, tab, identifier
                f.write(f"{result['smiles']}\t{predicted_type}_cell_wall\n")
            else:
                f.write("CC\tunknown_cell_wall\n")  # Fallback if no prediction
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{base_name}_prediction.txt")
        with open(results_file, 'w') as f:
            f.write(f"Cell Wall Prediction for {fastq_file}\n")
            f.write("="*50 + "\n\n")
            
            if predicted_type:
                f.write(f"Predicted cell wall type: {predicted_type}\n")
                f.write(f"Confidence: {confidence:.2%}\n\n")
            else:
                f.write("No prediction available (insufficient data)\n\n")
            
            f.write("Cell wall type distribution:\n")
            for cell_type in sorted(type_percentages.keys()):
                f.write(f"  {cell_type}: {type_percentages[cell_type]:.2f}% ({unique_genes_found[cell_type]} unique genes found)\n")
            
            f.write("\nSMILES representation:\n")
            f.write(f"  {result['smiles']}\n\n")
            
            f.write("SMILES saved to: " + smiles_file + "\n")
        
        # Print summary of results
        print("\nCell Wall Prediction Results:")
        if predicted_type:
            print(f"  Predicted type: {predicted_type}")
            print(f"  Confidence: {confidence:.2%}")
        else:
            print("  No prediction available (insufficient data)")
            
        print("\nCell wall type distribution:")
        for cell_type in sorted(type_percentages.keys()):
            print(f"  {cell_type}: {type_percentages[cell_type]:.2f}% ({unique_genes_found[cell_type]} unique genes found)")
            
        print(f"\nSMILES output saved to: {smiles_file}")
        print(f"Detailed results saved to: {results_file}")
        
        return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict bacterial cell wall type and output SMILES')
    parser.add_argument('--input', required=True, help='Input FASTQ file')
    parser.add_argument('--output-dir', default='./smiles_output', help='Output directory')
    parser.add_argument('--max-reads', type=int, default=100000, help='Maximum reads to process')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CellWallSMILESPredictor(output_dir=args.output_dir)
    
    # Run prediction
    predictor.predict_cell_wall_type(args.input, max_reads=args.max_reads)


if __name__ == "__main__":
    main()
