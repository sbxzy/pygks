import csv
from numpy import array

class csv_reader:
    """This class is for easy csv file reading. has_header is True or False. While set to True, this utility will jump the
    first line of csv file, thinking it as the header.
    Parameters: filename (string), deli(charactor)
    filename is the file you want to read, which contains the data for machine learning. deli is the delimiter of the csv file."""
    the_reader = 0
    count = 0
    
    def __init__(self,filename,has_header = False,deli = ','):
        self.the_reader = csv.reader(open(filename,'r'),delimiter = deli)
        if has_header:
            self.the_reader.next()

    def get_all(self):
        """Return the complete data in the form of list of numpy arrays."""
        block_all = []
        for line in self.the_reader:
            block_all.append(array(line,'float'))
        return block_all

    def separate_label(self):
        """Similar to get_all() method, but the last element in each line is separated as labels."""
        block_all = []
        label_all = []
        for line in self.the_reader:
            label_all.append(float(line[-1]))
            line.pop()
            block_all.append(array(line,'float'))
        return block_all,label_all

def __dict_reverse(the_dict):
    all_values = list(set(the_dict.values()))
    result = {}
    for each_value in all_values:
        result[each_value] = []
    for each_key,each_value in the_dict.items():
        result[each_value].append(each_key)
    return result
