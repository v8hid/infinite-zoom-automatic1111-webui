"""
def calculate_interpolation_steps_goldenratio(self,original_size, target_size, steps):
        width, height = original_size
        target_width, target_height = target_size
        golden_ratio = (1 + 5 ** 0.5) / 2 - 1  # Approx. 0.618

        if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0 or steps <= 0:
            return None

        original_ratio = width / height
        scaling_steps = []
        for i in range(1, steps + 1):
            t = i / steps
            factor = 1 - golden_ratio * t
            new_width = width * factor + target_width * (1 - factor)
            new_height = height * factor + target_height * (1 - factor)

            floor_width, ceil_width = int(new_width // 1), int(new_width // 1 + 1)
            floor_height, ceil_height = int(new_height // 1), int(new_height // 1 + 1)

            floor_ratio = floor_width / floor_height
            ceil_ratio = ceil_width / ceil_height

            if abs(floor_ratio - original_ratio) < abs(ceil_ratio - original_ratio):
                new_width, new_height = floor_width, floor_height
            else:
                new_width, new_height = ceil_width, ceil_height

            scaling_steps.append((new_width, new_height))

        return scaling_steps


    def calculate_interpolation_steps_log(self, original_size, target_size, steps):
        width, height = original_size
        target_width, target_height = target_size

        if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0 or steps <= 0:
            return None

        original_ratio = width / height
        scaling_steps = []
        
        log_w_ratio = math.log(target_width / width) / steps
        log_h_ratio = math.log(target_height / height) / steps

        for i in range(1, steps):
            new_width = width * math.exp(i * log_w_ratio)
            new_height = height * math.exp(i * log_h_ratio)

            floor_width, ceil_width = int(new_width // 1), int(new_width // 1 + 1)
            floor_height, ceil_height = int(new_height // 1), int(new_height // 1 + 1)

            floor_ratio = floor_width / floor_height
            ceil_ratio = ceil_width / ceil_height

            if abs(floor_ratio - original_ratio) < abs(ceil_ratio - original_ratio):
                new_width, new_height = floor_width, floor_height
            else:
                new_width, new_height = ceil_width, ceil_height

            scaling_steps.append((new_width, new_height))

        # Add the last step that is one pixel away from the target size
        scaling_steps.append((target_width - 1, target_height - 1))

        return scaling_steps


    def calculate_interpolation_steps_exponential(self, original_size, target_size, steps,exponent=2):
        width, height = original_size
        target_width, target_height = target_size
        scaling_steps = []
        for i in range(1, steps + 1):
            t = i / steps
            factor = (1 - t) + t * (math.pow(t, exponent - 1))
            new_width = width * (1 - factor) + target_width * factor
            new_height = height * (1 - factor) + target_height * factor
            scaling_steps.append((math.floor(new_width), math.floor(new_height)))
        return scaling_steps
"""