# Miscellaneous imports
from os import getcwd, listdir
from os.path import join, isfile, isdir


class Condense_Reviews(object):
    """
    The Condense_Reviews class takes the raw data, which comes distributed among many different files and
    condenses it all into three separate files for easy consumption:
        - positive reviews (pos.txt)
        - negative reviews (neg.txt)
        - unlabeled reviews (unlab.txt)

    The class is initialized with two parameters:
        - desired path for output files
        - path to data directory, i.e., the "aclImdb" directory
    """
    def __init__(self, data_dir, output_dir):
        self.output_dir = output_dir
        self.data_dir = data_dir


    def condense(self):
        # Organize subdirectories by sentiment
        pos_subdirs = [join(self.data_dir, "train", "pos"),
                       join(self.data_dir, "test", "pos")]
        neg_subdirs = [join(self.data_dir, "train", "neg"),
                       join(self.data_dir, "test", "neg")]
        unlab_subdirs = [join(self.data_dir, "train", "unsup")]

        # Define dictionary of sentiments paired with associated subdirectories
        sent_dict = {"pos": pos_subdirs,
                     "neg": neg_subdirs,
                     "unlab": unlab_subdirs}

        # Condense all files from positive sentiment subdirectories
        for sentiment, subdirs in sent_dict.iteritems():
            # Define full path for output file
            full_fname = join(self.output_dir, sentiment + ".txt")
            with open(full_fname, "w") as output_file:
                for _dir in subdirs:
                    for _file in listdir(_dir):
                        for line in open(join(_dir, _file)):
                            output_file.write(line)
                            output_file.write("\n")


def run_condenser():
    # If data is not yet condensed, condense it
    if (not isfile("pos.txt")) and (not isfile("neg.txt")) and (not isfile("unlab.txt")):
        if isdir("aclImdb"):
            print "Gathering data..."
            # Initialize a Condense_Reviews object
            output_dir = getcwd()
            data_dir = join(getcwd(), "aclImdb")

            # Condense reviews
            condenser = Condense_Reviews(data_dir, output_dir)
            condenser.condense()
        else:
            print "The data directory (aclImdb) seems to be missing. Please download or move a copy to current directory."
    else:
        print "Data is already condensed. Performing preprocessing..."


if __name__ == "__main__":
    run_condenser()
