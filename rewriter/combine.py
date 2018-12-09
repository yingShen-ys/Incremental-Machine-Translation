import csv
import argparse

# csv_header = ["original sent", "rewritten sent", 'rewritten']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', dest='csv', action='store', type=str, help='csv file', required=True)
	parser.add_argument('--ja', dest='ja', action='store', type=str, help='JA file', required=True)
	parser.add_argument('--ja_new', dest='ja_new', action='store', type=str, help='ja_new file', required=True)
	parser.add_argument('--en_new', dest='en_new', action='store', type=str, help='en_new file', required=True)
	args = parser.parse_args()

	csv_file = args.csv
	ja_file = args.ja
	ja_file_new = args.ja_new
	en_file_new = args.en_new

	ja_new = []
	en_new = []
	ja_id = 0

	with open(ja_file, 'r') as f:
		ja_sents = f.readlines()

	with open(csv_file, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',')
		first_line = True
		for row in csv_reader:
			if first_line:
				print(f'Column names are {", ".join(row)}')
				first_line = False
			else:
				if int(row[2]) == 1: # rewritten == True
					ja_new.append(ja_sents[ja_id])
					ja_new.append(ja_sents[ja_id])
					en_new.append(row[0] + '\n')
					en_new.append(row[1] + '\n')
				else:
					ja_new.append(ja_sents[ja_id])
					en_new.append(row[0] + '\n')
					ja_id += 1	

	assert len(ja_new) == len(en_new)
					
	with open(ja_file_new, 'w') as f:
		f.write(''.join(ja_new))

	with open(en_file_new, 'w') as f:
		f.write(''.join(en_new))	



