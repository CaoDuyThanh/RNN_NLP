import csv
import itertools
import nltk

def LoadData(fileName):
    print ('Reading CSV file...')

    with open(fileName) as file:
        reader = csv.reader(file, skipinitialspace = True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SETENCE_START and SETENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print ('Reading CSV file...Done! ')
    print "Parsed %d sentences." % (len(sentences))

    return sentences