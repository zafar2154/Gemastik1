import json

# Definisikan aturan penggabungan kelas di sini
MERGE_RULES = {
    'Urobilinogen': {'Normal': 'Normal', 'default': 'Tidak Normal'},
    'Protein': {'Neg.': 'Normal', 'default': 'Tidak Normal'},
    'Ketone': {'Neg.': 'Normal', 'default': 'Tidak Normal'},
    'Bilirubin': {'Neg.': 'Normal', 'default': 'Tidak Normal'},
    'pH': {
        '5.0': 'Asam', '6.0': 'Asam',
        '6.5': 'Netral', '7.0': 'Netral', '7.5': 'Netral',
        '8.0': 'Basa', '8.5': 'Basa'
    },
    'Specific Gravity': {
        '1.000': 'Rendah', '1.005': 'Rendah',
        '1.010': 'Normal', '1.015': 'Normal', '1.020': 'Normal',
        '1.025': 'Tinggi', '1.030': 'Tinggi'
    }
}

INPUT_FILE = 'labels.json'
OUTPUT_FILE = 'labels_merged.json'

def merge_labels():
    print(f"Membaca label dari {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        original_labels = json.load(f)

    merged_labels = {}
    
    for filename, labels in original_labels.items():
        new_labels = labels.copy() # Salin label asli
        for param, rules in MERGE_RULES.items():
            if param in new_labels:
                original_value = new_labels[param]
                # Terapkan aturan penggabungan
                if original_value in rules:
                    new_labels[param] = rules[original_value]
                elif 'default' in rules:
                    new_labels[param] = rules['default']
        merged_labels[filename] = new_labels
        
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(merged_labels, f, indent=4)
        
    print(f"Proses selesai. Label yang telah digabungkan disimpan di {OUTPUT_FILE}")
    print("Contoh hasil penggabungan:")
    print(json.dumps(list(merged_labels.values())[0], indent=2))

if __name__ == '__main__':
    merge_labels()