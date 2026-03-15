You are tasked with implementing a function that retrieves the instance of a given plugin from a window object. The function should take a string representing the plugin's name and return the plugin instance if it exists, or None if it does not. The window object is simulated as a nested dictionary where the keys represent the names of the plugins and the values represent the plugin instances. Your function should be able to handle cases where the specified plugin does not exist in the window object.\n\nImplement the function `get_plugin_instance(window: dict, plugin_name: str) -> Optional[object]` where:\n- `window` is a dictionary with plugin names as keys and their respective instances as values.\n- `plugin_name` is a string representing the name of the plugin.\n\nReturn the corresponding plugin instance from the window if it exists, otherwise return None.


## Base MODEL
```python
from typing import Dict, Optional

def get_plugin_instance(window: Dict[str, object], plugin_name: str) -> Optional[object]:
    # Check if the plugin exists in the window
    if plugin_name in window:
        return window[plugin_name]
    else:
        return None
```

**Created Question**:
How can I create a function that checks if a given plugin is installed in a window object?

**Created Answer**:
To check if a given plugin is installed in a window object, you can use the following function:

```python
def is_plugin_installed(window: Dict[str, object], plugin_name: str) -> bool:
    # Check if the plugin exists in the window
    if plugin_name in window:
        return True
    else:
        return False
```

This function takes a dictionary `window` and a string `plugin_name` as input. It returns `True` if the specified plugin exists in the window object, and `False` otherwise. This can be useful for determining whether certain plugins are available for use in a window application.


Trained Model

```python
from typing import Dict, Optional

def get_plugin_instance(window: Dict[str, object], plugin_name: str) -> Optional[object]:
    # Check if the plugin exists in the window
    if plugin_name in window:
        return window[plugin_name]
    else:
        return None
```

The provided code snippet defines a function `get_plugin_instance` that takes a dictionary `window` and a string `plugin_name` as parameters. It checks if the specified plugin exists in the window using the `in` keyword. If the plugin exists, it returns the corresponding plugin instance; otherwise, it returns None. This solution handles cases where the specified plugin does not exist in the window object.
