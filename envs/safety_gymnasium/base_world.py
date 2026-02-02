import numpy as np

# If we use smaller values, Mujoco simulation will be unstable
class BaseWorld:
    def name_space_for_hidden_parameters(self):
        basic_parameters = ["damping_mult", "gravity_mult", "mass_mult", "inertia_mult", "friction_mult"]
        # Later, we can add more parameters here
        additional_parameters = []
        return basic_parameters + additional_parameters

    def sample_hidden_parameters(
            self, 
            fix_hidden_parameters: bool = False, 
            is_out_of_distribution: bool = False, 
            min_mult: float = 0.3,
            max_mult: float = 1.7,  
            out_side_param: str = "",
            max_param_bound: float = 2.5,
            min_param_bound: float = 0.15,
        ):
        parameters_name = self.name_space_for_hidden_parameters()
        features_offset = 1.0
        if fix_hidden_parameters:
            return {param: 1.0 for param in parameters_name}, features_offset
        
        parameters = {}
        features_offset *= (min_mult + max_mult) / 2
        for basic_param in parameters_name:
            # Parameters are sampled from a uniform distribution outside of the range [min_mult, max_mult]
            if is_out_of_distribution:
                parameters[basic_param] = np.random.uniform(min_param_bound, min_mult) if np.random.choice([0, 1]) == 0 else np.random.uniform(max_mult, max_param_bound)
            else:
                parameters[basic_param] = np.random.uniform(min_mult, max_mult)

        if is_out_of_distribution:
            for param in out_side_param:
                parameters[param] = np.random.uniform(max_mult, max_param_bound)

        return parameters, features_offset
        

    def hidden_parameters_for_env_config(self):
        pass

    def hidden_parameters_for_task_config(self):
        pass