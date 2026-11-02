"""
MaternalRisk AI - Integration & Validation Test Suite
======================================================
Tests the Streamlit app components, SVM prediction engine, and Azure AI Foundry
agent integration across all requirements.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_maternal_risk, VALID_RISK_CLASSES
from src.agent_client import FoundryAgentClient


def test_1_svm_prediction_engine():
    """Verify Risk Assessment SVM prediction engine."""
    print("\n--- Test 1: Direct SVM Prediction Engine ---")
    test_cases = [
        {"age": 25, "systolic_bp": 120, "diastolic_bp": 80, "blood_sugar": 7.0, "body_temp": 98.0, "heart_rate": 70, "expected": "low risk"},
        {"age": 35, "systolic_bp": 140, "diastolic_bp": 100, "blood_sugar": 15.0, "body_temp": 98.0, "heart_rate": 90, "expected": "high risk"},
    ]
    for i, tc in enumerate(test_cases, 1):
        expected = tc.pop("expected")
        result = predict_maternal_risk(**tc)
        assert result in VALID_RISK_CLASSES, f"Result {result} not in {VALID_RISK_CLASSES}"
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"  [PASS] Case {i} inputs -> SVM Output: '{result}' (Matched expected '{expected}')")
    print("Test 1 Passed: SVM prediction pipeline operates correctly.")


def test_2_agent_health_check():
    """Verify live agent server health check."""
    print("\n--- Test 2: Agent Server Connectivity & Health ---")
    client = FoundryAgentClient()
    is_healthy, msg = client.check_health()
    print(f"  Server URL: {client.host_url}")
    print(f"  Health Result: is_healthy={is_healthy}, message='{msg}'")
    assert is_healthy, f"Agent server health check failed: {msg}"
    print("Test 2 Passed: Agent server is online and reachable.")


def test_3_agent_conversational_turn():
    """Verify general conversation without tools."""
    print("\n--- Test 3: Agent General Conversational Turn ---")
    client = FoundryAgentClient()
    messages = [
        {"role": "user", "content": "What is normal systolic blood pressure during pregnancy?"}
    ]
    result = client.send_chat_message(messages, timeout=45)
    assert result["success"] is True, f"Agent call failed: {result.get('error')}"
    assert len(result["text"]) > 20, "Response text was too short"
    assert len(result["tool_calls"]) == 0, f"Expected 0 tool calls for general inquiry, got {len(result['tool_calls'])}"
    print(f"  [PASS] Assistant response preview: {result['text'][:100]}...")
    print(f"  [PASS] Tool calls executed: {len(result['tool_calls'])}")
    print("Test 3 Passed: Agent conversational responses function correctly.")


def test_4_and_5_six_measurements_invokes_svm_tool():
    """Verify that 6 measurements trigger assess_maternal_risk and return real SVM class."""
    print("\n--- Test 4 & 5: Six Measurements Trigger assess_maternal_risk Tool ---")
    client = FoundryAgentClient()
    measurements = {
        "age": 25,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "blood_sugar": 7.0,
        "body_temp": 98.0,
        "heart_rate": 70,
    }
    # Direct SVM prediction for comparison
    direct_svm_result = predict_maternal_risk(**measurements)

    prompt = (
        f"Please evaluate maternal risk for: age {measurements['age']}, "
        f"systolic bp {measurements['systolic_bp']}, diastolic bp {measurements['diastolic_bp']}, "
        f"blood sugar {measurements['blood_sugar']}, body temp {measurements['body_temp']}, "
        f"heart rate {measurements['heart_rate']}."
    )
    messages = [{"role": "user", "content": prompt}]
    result = client.send_chat_message(messages, timeout=60)

    assert result["success"] is True, f"Agent call failed: {result.get('error')}"
    tool_calls = result.get("tool_calls", [])
    assert len(tool_calls) >= 1, f"Expected at least 1 tool call, got {len(tool_calls)}"

    maternal_tool = None
    for tc in tool_calls:
        if tc.get("name") == "assess_maternal_risk":
            maternal_tool = tc
            break

    assert maternal_tool is not None, "assess_maternal_risk tool was not invoked!"
    tool_output = maternal_tool.get("output", "")
    print(f"  [PASS] Tool Invoked: '{maternal_tool.get('name')}'")
    print(f"  [PASS] Tool Arguments: {maternal_tool.get('arguments')}")
    print(f"  [PASS] Tool Output: '{tool_output}'")
    print(f"  [PASS] Direct SVM Result: '{direct_svm_result}'")

    # Verify tool output reflects the real SVM classification
    assert direct_svm_result.upper() in tool_output.upper(), (
        f"Direct SVM result '{direct_svm_result}' not found in tool output '{tool_output}'"
    )
    # Verify assistant text mentions the classification
    assert direct_svm_result.lower() in result["text"].lower() or direct_svm_result.upper() in result["text"].upper(), (
        f"Classification '{direct_svm_result}' not reflected in assistant text"
    )
    print("Test 4 & 5 Passed: Actual assess_maternal_risk tool executed and returned real SVM classification.")


def test_6_graceful_unavailable_agent_handling():
    """Verify application handles an unavailable agent gracefully without crashing."""
    print("\n--- Test 6: Graceful Handling When Agent is Offline ---")
    offline_client = FoundryAgentClient(host_url="http://localhost:59999")
    is_healthy, status_msg = offline_client.check_health(timeout=1)
    assert is_healthy is False, "Expected offline client health check to return False"
    print(f"  [PASS] Health check offline detection: {status_msg}")

    result = offline_client.send_chat_message([{"role": "user", "content": "Hello"}], timeout=2)
    assert result["success"] is False, "Expected offline send_chat_message to return success=False"
    assert result["error_type"] == "connection", f"Expected error_type='connection', got {result.get('error_type')}"
    assert "Unable to connect" in result["error"], "Expected informative error message"
    print(f"  [PASS] Error handled gracefully: error_type='{result['error_type']}'")
    print("Test 6 Passed: Offline agent handled cleanly without crashing.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MATERNALRISK AI INTEGRATION TEST SUITE")
    print("=" * 70)
    test_1_svm_prediction_engine()
    test_2_agent_health_check()
    test_3_agent_conversational_turn()
    test_4_and_5_six_measurements_invokes_svm_tool()
    test_6_graceful_unavailable_agent_handling()
    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
