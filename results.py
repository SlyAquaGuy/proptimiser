from init import self
## Write Data to File


# Save Solver Logs to .txt file
if self.save_solver_outputs:
    with open(self.save_file_name, 'w') as f:
        for key, value in self.solver_logs.items():
            f.write(f"{key}: {value}\n")

print()