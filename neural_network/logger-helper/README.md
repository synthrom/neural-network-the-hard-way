# Logger Helper

Create a simple Python logger

## Installation
#### New install
HTTPS: pip install git+https://code.osu.edu/enterprise-security/utilities/logger-helper.git  
SSH:   pip install git+ssh://git@code.osu.edu/enterprise-security/utilities/logger-helper.git

#### Upgrade

HTTPS: pip install git+https://code.osu.edu/enterprise-security/utilities/logger-helper.git --upgrade  
SSH:   pip install git+ssh://git@code.osu.edu/enterprise-security/utilities/logger-helper.git --upgrade

## Usage
Easily make a logger for your script!
``` python
from create_logger import create_logger
logger = create_logger(
    name='splunkdata', 
    loggerLevel='debug', 
    consoleHandlerLevel='warning', 
    fileHandlerLevel='error'
)
logger.debug('This is a debug message')
```
for more info: https://docs.python.org/3/library/logging.html#logging-levels
