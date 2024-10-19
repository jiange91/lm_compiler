from . import candidate_generation
from . import column_filtering
from . import column_selection
from . import keyword_extraction
from . import revision
from . import table_selection

_cognify_lm_registry = {
    'keyword_extraction': keyword_extraction.exec,
    'column_filtering': column_filtering.exec,
    'table_selection': table_selection.exec,
    'column_selection': column_selection.exec,
    'candidate_generation': candidate_generation.exec,
    'revision': revision.exec  
}