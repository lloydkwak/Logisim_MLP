import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import struct

#데이터 전처리
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    # 이진화: 0.7 임계값
    transforms.Lambda(lambda x: (x > 0.7).float())
])

#데이터셋 및 로더
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

#32비트 부동소수점 최적화 MLP 모델
class OptimizedFloat32MLP(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        #가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        #He 초기화(ReLU에 최적화)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        
        #바이어스 초기화
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)  #출력층에는 활성화 함수 없음
        return x

#32비트 부동소수점 강제 적용
def ensure_float32(model):
    for param in model.parameters():
        param.data = param.data.float()

#학습
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #32비트 부동소수점으로 강제 변환
        data, target = data.float().to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        #그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

#테스트
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test accuracy: {acc:.2f}%")
    return acc

#로지심 주소 매핑
def logisim_address_to_python_index(logisim_addr):
    y = (logisim_addr >> 4) & 0xF  #상위 4비트
    x = logisim_addr & 0xF         #하위 4비트
    return y * 16 + x

def python_index_to_logisim_address(python_index):
    y = python_index // 16
    x = python_index % 16
    return (y << 4) | x

def create_logisim_address_mapping():
    mapping = []
    for logisim_addr in range(256):
        python_idx = logisim_address_to_python_index(logisim_addr)
        mapping.append(python_idx)
    return mapping

#RAM 지연 보정에 맞게 가중치 시프트
def shift_weights_for_ram_delay(weights_array):
    if len(weights_array) == 0:
        return weights_array
    
    shifted = np.zeros_like(weights_array)
    shifted[0] = weights_array[-1]
    shifted[1:] = weights_array[:-1] 
    
    return shifted

#로지심 주소 매핑에 맞춘 가중치 저장
def save_weights_float32_logisim_mapped_with_shift(model, output_dir='logisim_mapped_weights_shifted'):
    os.makedirs(output_dir, exist_ok=True)
    
    #가중치 및 바이어스 추출
    fc1_weights = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    fc1_bias = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    fc2_weights = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
    fc2_bias = model.fc2.bias.detach().cpu().numpy().astype(np.float32)
    
    print("=== RAM 지연 보정 32비트 부동소수점 가중치 통계 ===")
    print(f"FC1 weights - 평균: {fc1_weights.mean():.6f}, 표준편차: {fc1_weights.std():.6f}")
    print(f"FC1 weights - 범위: {fc1_weights.min():.6f} ~ {fc1_weights.max():.6f}")
    print(f"FC1 bias - 평균: {fc1_bias.mean():.6f}, 표준편차: {fc1_bias.std():.6f}")
    print(f"FC2 weights - 평균: {fc2_weights.mean():.6f}, 표준편차: {fc2_weights.std():.6f}")
    print(f"FC2 bias - 평균: {fc2_bias.mean():.6f}, 표준편차: {fc2_bias.std():.6f}")
    
    #로지심 주소 매핑 테이블
    address_mapping = create_logisim_address_mapping()
    
    #IEEE 754 32비트 부동소수점을 16진수로 변환
    def float32_to_hex_string(value):
        """32비트 부동소수점을 IEEE 754 16진수로 변환"""
        return format(struct.unpack('>I', struct.pack('>f', value))[0], '08x')
    
    #FC1 가중치 저장
    layer1_weights_path = os.path.join(output_dir, "layer1_weights_shifted.txt")
    with open(layer1_weights_path, 'w') as f:
        f.write("v2.0 raw\n")
        hex_values = []
        
        #각 은닉 뉴런(128개)에 대해
        for hidden_neuron in range(fc1_weights.shape[0]):
            #로지심 주소 순서대로 가중치 추출
            neuron_weights = []
            for logisim_addr in range(256):
                input_idx = address_mapping[logisim_addr]
                neuron_weights.append(fc1_weights[hidden_neuron, input_idx])
            
            #RAM 지연 보정을 위한 시프트
            shifted_weights = shift_weights_for_ram_delay(neuron_weights)
            
            #시프트된 가중치를 16진수로 변환하여 저장
            for weight_val in shifted_weights:
                hex_val = float32_to_hex_string(weight_val)
                hex_values.append(hex_val)
                
                if len(hex_values) % 8 == 0:
                    f.write(" ".join(hex_values) + "\n")
                    hex_values = []
        
        if hex_values:
            f.write(" ".join(hex_values) + "\n")
    
    #FC2 가중치 저장
    layer2_weights_path = os.path.join(output_dir, "layer2_weights_shifted.txt")
    with open(layer2_weights_path, 'w') as f:
        f.write("v2.0 raw\n")
        hex_values = []
        
        #각 출력 뉴런(10개)에 대해
        for output_neuron in range(fc2_weights.shape[0]):
            #은닉층 가중치 추출
            neuron_weights = fc2_weights[output_neuron, :]
            
            #RAM 지연 보정을 위한 시프트
            shifted_weights = shift_weights_for_ram_delay(neuron_weights)
            
            #시프트된 가중치를 16진수로 변환하여 저장
            for weight_val in shifted_weights:
                hex_val = float32_to_hex_string(weight_val)
                hex_values.append(hex_val)
                
                if len(hex_values) % 8 == 0:
                    f.write(" ".join(hex_values) + "\n")
                    hex_values = []
        
        if hex_values:
            f.write(" ".join(hex_values) + "\n")
    
    #FC1 바이어스 저장
    bias1_path = os.path.join(output_dir, "layer1_bias_shifted.txt")
    with open(bias1_path, 'w') as f:
        f.write("v2.0 raw\n")
        
        #바이어스 시프트
        shifted_bias = shift_weights_for_ram_delay(fc1_bias)
        
        hex_values = []
        for bias_val in shifted_bias:
            hex_val = float32_to_hex_string(bias_val)
            hex_values.append(hex_val)
            if len(hex_values) % 8 == 0:
                f.write(" ".join(hex_values) + "\n")
                hex_values = []
        if hex_values:
            f.write(" ".join(hex_values) + "\n")
    
    #FC2 바이어스 저장
    bias2_path = os.path.join(output_dir, "layer2_bias_shifted.txt")
    with open(bias2_path, 'w') as f:
        f.write("v2.0 raw\n")
        
        #바이어스 시프트
        shifted_bias = shift_weights_for_ram_delay(fc2_bias)
        
        hex_values = []
        for bias_val in shifted_bias:
            hex_val = float32_to_hex_string(bias_val)
            hex_values.append(hex_val)
            if len(hex_values) % 8 == 0:
                f.write(" ".join(hex_values) + "\n")
                hex_values = []
        if hex_values:
            f.write(" ".join(hex_values) + "\n")
    
    return {
        'fc1_weights': fc1_weights,
        'fc1_bias': fc1_bias,
        'fc2_weights': fc2_weights,
        'fc2_bias': fc2_bias,
    }

def main():
    #32비트 부동소수점 강제 설정
    torch.set_default_dtype(torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    print(f"기본 텐서 타입: {torch.get_default_dtype()}")
    
    #MLP 모델 생성
    model = OptimizedFloat32MLP(
        input_size=16*16, 
        hidden_size=128, 
        output_size=10
    ).to(device)
    
    #32비트 부동소수점 강제 적용
    ensure_float32(model)
    
    #최적화된 학습률과 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("학습 전 정확도:")
    test(model, device, test_loader)
    
    best_acc = 0
    #에폭 수
    for epoch in range(1, 20):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        
        #학습률 조정
        scheduler.step()
        
        #최고 정확도 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model_ram_shift.pth')
    
    #최고 정확도 모델 로드
    model.load_state_dict(torch.load('best_model_ram_shift.pth'))
    
    # 최종 정확도 테스트
    print("\n=== 최종 RAM 지연 보정 모델 정확도 ===")
    final_acc = test(model, device, test_loader)
    
    #가중치 저장
    save_weights_float32_logisim_mapped_with_shift(model)
    
    print(f"\n최종 결과:")
    print(f"RAM 지연 보정 모델 정확도: {final_acc:.2f}%")
    print(f"시프트된 가중치 파일 생성 완료")

if __name__ == "__main__":
    main()
