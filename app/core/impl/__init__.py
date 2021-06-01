from .predict import predict
from .load_specified_sample import load_specified_sample
from .load_sample import load_sample
from .patients import (
    get_patients,
    get_patient,
    import_patient,
)
from .favorites import (
    get_favorites,
    add_favorite,
    modify_favorite,
    delete_favorites,
)
from .ner import import_ner
from .login import login
from .user import (
    get_users,
    update_user
    )
from .register import register
from .similarity import similarity
