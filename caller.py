from framework.tasks.arc_dataset import load_task
from framework.tasks.base import ArcTask
from framework.grids import pretty_grid

# Pick any ARC id present in arc_original.zip, e.g. "a85d4709"
task_id = "a85d4709"

task: ArcTask = load_task(task_id)

print("Task ID:", task.task_id)
print("Num train pairs:", len(task.train_pairs))
print("Num test inputs:", len(task.test_inputs))

print("\nFirst train input grid:")
print(pretty_grid(task.train_pairs[0].input))
print("\nFirst train output grid:")
print(pretty_grid(task.train_pairs[0].output))

print("\nHas re_arc synthetic pairs?", task.re_arc_synthetic_pairs is not None)
if task.re_arc_synthetic_pairs is not None:
    print("  Num re_arc synthetic pairs:", len(task.re_arc_synthetic_pairs))
    print("  First re_arc pair:", task.re_arc_synthetic_pairs[0])
print("Has arc_gen synthetic pairs?", task.arc_gen_synthetic_pairs is not None)
if task.arc_gen_synthetic_pairs is not None:
    print("  Num arc_gen synthetic pairs:", len(task.arc_gen_synthetic_pairs))
    print("  First arc_gen pair:", task.arc_gen_synthetic_pairs[0])
print("Has re_arc generator?", task.re_arc_generator is not None)
if task.re_arc_generator is not None:
    print("  Sample from re_arc generator (1 example):", task.re_arc_generator(1)[0])
print("Has arc_gen generator?", task.arc_gen_generator is not None)
if task.arc_gen_generator is not None:
    print("  Sample from arc_gen generator (1 example):", task.arc_gen_generator(1)[0])
print("Has primary verifier?", task.verifier is not None)
if task.verifier is not None:
    print("  Verifier:", task.verifier)
    print("  Verifier output on first train input:",
          task.verifier(task.train_pairs[0].input))
print("Has secondary verifier?", task.secondary_verifier is not None)
if task.secondary_verifier is not None:
    print("  Secondary verifier:", task.secondary_verifier)
    print("  Secondary output on first train input:",
          task.secondary_verifier(task.train_pairs[0].input))