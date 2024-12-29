from pydantic import BaseModel, ConfigDict
from typing import List

class YearQueryResponse(BaseModel):
    """
    Describes the layout of the JSON response to be returned when asking for year information.  The properties are case sensitive.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    YEAR: int
    """ The year of the report """

class GeneralInfoQueryResponse(BaseModel):
    """
    Describes the layout of the JSON response to be returned when asking for general information about the company.  The properties are case sensitive.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    COMPANY_NAME: str
    """ The name of the company """
    COMPANY_SECTOR: str
    """ The sector of the company """
    COMPANY_LOCATION: str
    """ The location of the company """

class QueryResponse(BaseModel):
    """
    Describes the layout of the JSON response to be returned when asking specific questions about the report.  The properties are case sensitive.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    ANSWER: bool
    """ use answer true if you would answer the question with yes.  answer false if you would anser the question with no  """
    EXPLANATION: str
    """ thorough explanation of why you judged the answer should be judged as yes or no.  sources should not be enumerated here, these should be enumerated in the SOURCES field """
    SOURCES: List[int]
    """ a list of the SOURCE numbers that were referenced in your answer """
