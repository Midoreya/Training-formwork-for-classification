import torch

def export_onnx(model,
                input_size=224,
                name='test',
                opset_version=11,
                Simplify=True,
                root='./'):

    if not root.endswith('/'):
        root = root + '/'
    fake_input = torch.zeros(1, 3, input_size, input_size)

    torch.onnx.export(model,
                      fake_input,
                      root + name + '.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=opset_version
    )
    
    if Simplify == True:
        import onnx
        from onnxsim import simplify
        
        onnx_model = onnx.load(root + name + '.onnx')
        onnx_sim_model, check = simplify(onnx_model)
        onnx.save(onnx_sim_model, root + name + '.onnx')
        
    print('Export onnx model as:\n', root + name + '.onnx\n')

def export_jit(model,
               input_size=224,
               name='test',
               root='./'):
    
    if not root.endswith('/'):
        root = root + '/'
    fake_input = torch.zeros(1, 3, input_size, input_size)
    
    s_net = torch.jit.trace(model, fake_input)
    s_net.save(root + name + '.pt')
    
    print('Export jit model as:\n', root + name + '.pt\n')
    
    
    