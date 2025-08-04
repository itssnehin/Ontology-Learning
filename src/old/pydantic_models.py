from typing import List, Optional
from pydantic import BaseModel, Field

class Property(BaseModel):
    name: str = Field(description="Name of the property")
    value: str = Field(description="Value of the property, including units if applicable")

class Relationship(BaseModel):
    type: str = Field(description="Type of relationship (e.g., 'compatibleWith', 'interfacesWith')")
    target: str = Field(description="Target component or concept name")
    description: Optional[str] = Field(default=None, description="Optional description of the relationship")

class Application(BaseModel):
    name: str = Field(description="Name of the application domain")
    description: Optional[str] = Field(default=None, description="Brief description of the application")

class Component(BaseModel):
    component_type: str = Field(description="General classification of the component")
    model_number: str = Field(description="Specific model number of the component")
    manufacturer: str = Field(description="Company that produces the component")
    properties: List[Property] = Field(description="List of technical properties")
    features: List[str] = Field(description="Key features of the component")
    applications: Optional[List[Application]] = Field(
        default=[],
        description="Application domains where this component is used"
    )
    relationships: Optional[List[Relationship]] = Field(
        default=[],
        description="Relationships to other components or concepts"
    )
