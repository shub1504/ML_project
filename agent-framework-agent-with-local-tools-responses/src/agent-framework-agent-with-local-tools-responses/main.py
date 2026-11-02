
import os

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import Annotated

from maternal_predict import predict_maternal_risk

load_dotenv()


@tool(approval_mode="never_require")
def assess_maternal_risk(
    age: Annotated[float, Field(description="Age in years")],
    systolic_bp: Annotated[float, Field(description="Systolic blood pressure in mmHg")],
    diastolic_bp: Annotated[float, Field(description="Diastolic blood pressure in mmHg")],
    blood_sugar: Annotated[float, Field(description="Blood sugar measurement using the same unit and convention as the model training data")],
    body_temp: Annotated[float, Field(description="Body temperature in Fahrenheit, matching the model's training convention")],
    heart_rate: Annotated[float, Field(description="Heart rate in beats per minute")],
) -> str:
    """Run the existing experimental MaternalRisk SVM model on six supplied measurements."""

    try:
        result = predict_maternal_risk(
            age=age,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            blood_sugar=blood_sugar,
            body_temp=body_temp,
            heart_rate=heart_rate,
        )

        return (
            f"Experimental SVM classification: {result.upper()}. "
            "This is a research-model output, not a clinical diagnosis. "
            "It does not establish whether a pregnancy is safe or unsafe. "
            "A qualified healthcare professional must assess individual health concerns."
        )

    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return f"Prediction could not be completed: {exc}"


MATERNALRISK_INSTRUCTIONS = """
You are MaternalRisk AI, an educational maternal-health assistant.

You can explain maternal-health concepts and, when all six required
measurements are supplied, use the assess_maternal_risk tool to obtain
the actual output from the project's experimental SVM model.

Never invent or estimate a model prediction. Never claim to have run
the model unless the tool returns a result.

The model uses six inputs:
Age, systolic blood pressure, diastolic blood pressure, blood sugar,
body temperature and heart rate.

Explain the returned classification clearly. Do not invent probability
or confidence scores. The model is experimental and has not been
clinically validated. Its output is not a diagnosis and must never
replace professional assessment.

Never prescribe medication, recommend changing treatment or reassure
a user that they are safe based on a model prediction.

If a user describes severe bleeding, seizures, difficulty breathing,
fainting, severe headache with vision changes, or severe abdominal pain,
advise immediate medical attention. Do not delay emergency care to run
a prediction.

Do not request identifying patient information. Use empathetic,
simple language and explain technical terms.
"""


def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )

    agent = Agent(
        client=client,
        instructions=MATERNALRISK_INSTRUCTIONS,
        tools=[assess_maternal_risk],
        default_options={"store": False},
    )

    server = ResponsesHostServer(agent)
    server.run()


if __name__ == "__main__":
    main()
