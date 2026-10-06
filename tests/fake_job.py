"""Fake job speaking the Training Hub protocol (python -m tests.fake_job). params: steps, delay, fail."""
import time

from src.jobs import proto


def run(p, run_dir):
    for i in range(int(p["steps"])):
        print(f"step {i}")
        proto.progress("fake", (i + 1) / p["steps"] * 100, epoch=i + 1, epochs=p["steps"], loss=1 / (i + 1))
        time.sleep(p["delay"])
        if p["fail"] and i == 1:
            raise RuntimeError("boom")
    proto.metric("ic", 0.5)
    return {"model": "model/x"}


if __name__ == "__main__":
    proto.main("fake", run, defaults={"steps": 5, "delay": 0.05, "fail": 0})
