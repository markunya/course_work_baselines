import inspect
from utils.class_registry import ClassRegistry

losses_registry = ClassRegistry()

class LossBuilder:
    def __init__(self, config):
        self.losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config['losses'].items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if 'losses_args' in config and loss_name in config['losses_args']:
                loss_args = config['losses_args']['loss_name']
            self.losses[loss_name] = losses_registry[loss_name](**loss_args)

    def calculate_loss(self, info, tl_suffix=None):
        loss_dict = {}
        total_loss_str = 'total_loss'
        if tl_suffix is not None:
            total_loss_str += '_' + tl_suffix
        loss_dict[total_loss_str] = 0.0

        calculated_losses_set = set()

        for loss_name, loss in self.losses.items():
            signature = inspect.signature(loss.forward)
            param_names = [param.name for param in signature.parameters.values()]

            current_loss = 0.0
            for suffix, kwargs in info.items():
                if suffix != '':
                    suffix = '_' + suffix

                loss_args = {param: kwargs[param] for param in param_names if param in kwargs}
                if len(loss_args) < len(param_names):
                    continue
                
                loss_val = loss(**loss_args)
                calculated_losses_set.add(loss_name)
                current_loss += loss_val
                loss_dict[loss_name + suffix] = float(loss_val)
    
            loss_dict[total_loss_str] += self.coefs[loss_name] * current_loss

        not_calculated_losses = list(set(self.losses.keys()).difference(calculated_losses_set))
        if len(not_calculated_losses) > 0:
            raise RuntimeWarning(f'Losses {not_calculated_losses} from config was not calculated.'+
                                'Possibly beacuse of was not given enough arguments for that.')

        return loss_dict[total_loss_str], loss_dict