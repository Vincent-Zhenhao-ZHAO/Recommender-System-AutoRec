# Make font colours in terminal
# Reference: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
class bcolours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'


    REMOVE_PREV = '\033[1A\033[K'

 
    line_counter = 1

    @staticmethod
    def lincr(incr = 1): 
        bcolours.line_counter += incr

    def remove_lincr(): 
        print(bcolours.REMOVE_PREV* bcolours.line_counter)
        bcolours.line_counter = 1
         