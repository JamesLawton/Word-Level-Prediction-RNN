with open('input_data/Book 7 - The Deathly Hallows_djvu.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [line for line in content if (not line.startswith('Page |')) and (not line == '\n')]
print(content)
output_file = open('preprocessed_hp.txt', 'w')
for line in content:
  output_file.write(line.replace('\n', ''))
output_file.close()