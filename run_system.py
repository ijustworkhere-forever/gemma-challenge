import subprocess
import yaml
import time

def load_config():
    with open("system_config.yaml", "r") as f:
        return yaml.safe_load(f)


def start(cmd):
    return subprocess.Popen(cmd, shell=True)


if __name__ == "__main__":

    cfg = load_config()

    processes = []

    print("BOOTING RESEARCH SYSTEM\n")

    if cfg["system"]["mode"] == "single-node":

        processes.append(start("python swarm_leaderboard_server.py"))
        time.sleep(1)

        processes.append(start("python distributed_research_swarm.py"))
        time.sleep(1)

        processes.append(start("python autonomous_research_agent.py"))

        if cfg["dashboard"]["enabled"]:
            processes.append(start("streamlit run research_autopilot_dashboard.py"))

    print("\nSYSTEM ONLINE")
    print("Processes:", len(processes))

    for p in processes:
        p.wait()
