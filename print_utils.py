#################
##### usage #####
#################
# from print_utils import set_print_output_file, get_print_func
# oprint = print
# print = get_print_func()
# printOutFile = None

# printOutFile = 'output.txt'
# print = set_print_output_file(printOutFile)
# print('Saving print output to:')
# print(printOutFile)
# oprint('Not saved output')
                
def super_print(filename, encoding='cp1252'):
    '''filename is the file where output will be written'''
    def wrap(func):
        '''func is the function you are "overriding", i.e. wrapping'''
        def wrapped_func(*args,**kwargs):
            '''*args and **kwargs are the arguments supplied to the overridden function'''
            #use with statement to open, write to, and close the file safely
            try:
                with open(filename,'a', encoding=encoding) as outputfile:
                    if len(args) == 0:
                        args1=('\n',)
                    else:
                        if len(str(args[0])) > 0:
                            if type(args[0]) == type(''):
                                if args[0][0] == '\r':
                                    args1 = ('\n'+args[0][1:], ) + args[1:]
                                else:
                                    args1 = ('\n'+str(args[0]), ) + args[1:]
                            else:
                                args1 = ('\n', ) + args
                        else:
                            args1 = ('\n', ) + args[1:]

                    if len(args1) == 1:
                        outputfile.write(*args1) #,**kwargs)
                    else:
                        for arg in args1:
                            outputfile.write(*(str(arg)+'\t',)) # ,**kwargs)
            except Exception as e:
                pass
                func(e)
            #now original function executed with its arguments as normal
            return func(*args,**kwargs)
        return wrapped_func
    return wrap

printOutFile = None # 'output.txt'
oprint = print
print = super_print(printOutFile)(print)

def set_print_output_file(filename=None, encoding='cp1252'):
    global printOutFile
    global print 

    if filename is None:
        print = oprint
        printOutFile = None
    else:
        printOutFile = filename
        print = super_print(printOutFile, encoding=encoding)(oprint)

    return print

def get_print_func():
    if printOutFile is None:
        return oprint
    else:
        return print
    

