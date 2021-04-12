import re, sys, functools, traceback

class Utils:
#-------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def tryit(continueExcution=True):
        def decorator_repeat(func):
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                    return value
                except Exception as ex:
                    Utils.log(0,'ERROR: {}', Utils.formatError(ex, func.__name__))
                    return continueExcution if (continueExcution == True) else sys.exit(0)
            return wrapper_decorator
        return decorator_repeat
    
    @staticmethod
    def log(level, message, *argv):
        start, tab, newline, sub = '+ ','\t','\n','|__'
        form = newline+start if(level == 0) else (tab*level)+sub
        print(form + message.format(*argv))
    
    @staticmethod
    def formatError(e, inFunc = None):
        if inFunc != None:
            trace = traceback.format_exc()
            pyFile, lineNo = list(re.compile('File "(.*)".* line ([0-9]+),.*{}'.format(inFunc),re.MULTILINE).findall(trace)[0])
            return '{} | {} | {}:{} | {}'.format(str(e), pyFile, inFunc, lineNo, str(type(e)).replace('<','[').replace('>',']'))

        exc_type, exc_obj, exc_tb = sys.exc_info()
        pyname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return str(e).replace("\n", ", ") + " | " + pyname +":" + str(exc_tb.tb_lineno) +" | " + str(type(e))
#-----------------------------------------------------------------------------------------------------------------------