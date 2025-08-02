import pytest
from openai import OpenAI
import time
from transformers import AutoConfig


# Test parameter processing
def pytest_addoption(parser):
    parser.addoption("--model", action="store", required=True, help="Model to be tested")
    parser.addoption("--model-path", action="store", default="", help="Model path")
    parser.addoption("--max-model-len", action="store", type=int, default=4096, help="MML")
    parser.addoption("--host", action="store", default="localhost", help="Host to be tested")
    parser.addoption("--port", action="store", default="8000", help="Port to be tested")
    parser.addoption(
        "--engine", 
        action="store", 
        default="sglang", 
        choices=["sglang", "llm", "vllm"],
        help="Engine to be tested"
    )
    parser.addoption(
        "--use-docker",
        action="store_true",
        help="Use vllm docker image deployed.",
    )
    parser.addoption("--fim-template", action="store", default="", help="fim template for coder model")


@pytest.fixture(scope="session")
def model(pytestconfig):
    return pytestconfig.getoption("model")

@pytest.fixture(scope="session")
def host(pytestconfig):
    return pytestconfig.getoption("host")

@pytest.fixture(scope="session")
def port(pytestconfig):
    return pytestconfig.getoption("port")

@pytest.fixture(scope="session")
def engine(pytestconfig):
    return pytestconfig.getoption("engine")

@pytest.fixture(scope="session")
def max_model_len(pytestconfig):
    return pytestconfig.getoption("max_model_len")

@pytest.fixture(scope="session")
def use_docker(pytestconfig):
    return pytestconfig.getoption("--use-docker")

@pytest.fixture(scope="session")
def model_config(pytestconfig):
    model_path = pytestconfig.getoption("model_path")
    if model_path:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config
    else:
        return None

@pytest.fixture(scope="session")
def fim_template(pytestconfig):
    return pytestconfig.getoption("fim_template")

@pytest.fixture(scope="session")
def client(host, port):
    openai_cli = OpenAI(
        api_key="sk-das5uesnxk5skugn",  # Dedicated key for functional automation
        base_url=f"https://cloud.llm-ai.com/vllm/v1/"
    )
    return openai_cli


# Report processing
@pytest.hookimpl(tryfirst=True)
def pytest_html_results_table_header(cells):
    cells.insert(1, '<th class="sorttable" data-column-type="priority">Priority</th>')
    cells.insert(2, '<th class="sorttable" data-column-type="description">Description</th>')


@pytest.hookimpl(tryfirst=True)
def pytest_html_results_table_row(report, cells):
    cells.insert(1, f'<td class="col-priority">{report.priority}</td>')
    cells.insert(2, f'<td class="col-description">{report.description}</td>')
    link_text = "View Test Cases"
    link_url = "todo"
    cells[5] = f'<td class="col-links"><a href="{link_url}" target="_blank">{link_text}</a></td>'

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    description = ""
    marker = item.get_closest_marker("description")
    if marker:
        description = marker.args[0]
    report.description = description

    priority = ""
    for mark in ["P0", "P1", "P2"]:
        if item.get_closest_marker(mark):
            priority = mark
            break
    report.priority = priority

