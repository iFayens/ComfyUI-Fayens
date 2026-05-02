import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODES_DIR = os.path.join(os.path.dirname(__file__), "nodes")

for filename in os.listdir(NODES_DIR):
    if filename.endswith(".py") and not filename.startswith("_"):
        module_name = filename[:-3]
        module = importlib.import_module(f".nodes.{module_name}", package=__name__)

        # Parcourt toutes les classes du module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # On filtre les classes avec les attributs ComfyUI
            if hasattr(attr, "INPUT_TYPES") and hasattr(attr, "FUNCTION"):
                
                class_name = attr_name
                NODE_CLASS_MAPPINGS[class_name] = attr

                # Nom affiché custom si défini dans la classe
                display_name = getattr(attr, "DISPLAY_NAME", class_name)
                NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name