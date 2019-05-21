#!/usr/bin/env python
from resource_management import *

# server configurations
config = Script.get_config()

#smoke_test_user = config['configurations']['lustrefs-config-env']['dummy_user']
smoke_test_user = 'root'