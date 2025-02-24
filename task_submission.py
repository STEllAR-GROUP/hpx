import time

def submit_task(task_name):
    print(f"Task '{task_name}' is being submitted.")
    time.sleep(2)  # Simulating task processing time
    print(f"Task '{task_name}' is complete.")

def main():
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        submit_task(task)

if __name__ == "__main__":
    main()
