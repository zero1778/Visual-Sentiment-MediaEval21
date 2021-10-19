from torch.autograd import Function, Variable

class ResNetWSL(nn.Module):
    
    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_ftrs = model.fc.in_features

        self.downconv = nn.Sequential(
            nn.Conv2d(2048, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.GAP = nn.AvgPool2d(14)
        self.GMP = nn.MaxPool2d(14)
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x_ori = x  
        # detect branch
        x = self.downconv(x) 
        x_conv = x              
        x = self.GAP(x)  #x = self.GMP(x)       
        x = self.spatial_pooling(x) 
        x = x.view(x.size(0), -1)
        # cls branch
        x_conv = self.spatial_pooling(x_conv) 
        x_conv = x_conv * x.view(x.size(0),x.size(1),1,1) 
        x_conv = self.spatial_pooling2(x_conv) 
        x_conv_copy = x_conv
        for num in range(0,2047):            
            x_conv_copy = torch.cat((x_conv_copy, x_conv),1) 
        x_conv_copy = torch.mul(x_conv_copy,x_ori)
        x_conv_copy = torch.cat((x_ori,x_conv_copy),1) 
        x_conv_copy = self.GAP(x_conv_copy)
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0),-1)
        x_conv_copy = self.classifier(x_conv_copy)
        return x, x_conv_copy

class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps)(input)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling2 = nn.Sequential()
        pooling2.add_module('class_wise', ClassWisePool(num_classes))
        model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)
        
        input_size = 448

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)