import sys
from resource_management import *

class DummyServiceCheck(Script):

    def service_check(self, env):
    	import params
        print 'Service Check'
        Execute( "ls -la",
            tries     = 3,
            try_sleep = 5,
            logoutput = True
        )

if __name__ == "__main__":
  DummyServiceCheck().execute()