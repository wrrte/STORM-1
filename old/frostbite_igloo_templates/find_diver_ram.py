import numpy as np

def main():
    demo_path = '/media/storage_data/ai2lab/choemj/Drama_modified/demonstrations/Seaquest_20260729_225940.npz'
    print(f"Loading {demo_path}...")
    data = np.load(demo_path)
    ram = data['ram']  # (N, 128)
    
    # 1. 하단 잠수부 수(RAM[62])가 증가하는 프레임 찾기 (잠수부 획득 순간)
    diver_count = ram[:, 62]
    pickup_frames = np.where(np.diff(diver_count) == 1)[0]
    
    print(f"총 {len(pickup_frames)}번의 잠수부 획득(Pickup) 순간을 찾았습니다.")
    
    # AtariARI 논문에 따르면 Submarine X=70, Y=97
    sub_x = ram[:, 70]
    sub_y = ram[:, 97]
    
    # 2. 획득 직전(pickup_frames)에 Submarine의 X, Y 좌표와 가장 비슷한 값을 가졌던 RAM 주소 찾기
    # 잠수부를 획득하려면 잠수함과 화면상 잠수부의 위치가 겹쳐야 합니다.
    # 따라서 획득 직전 프레임에서 특정 RAM 주소의 값이 sub_x 또는 sub_y와 매우 유사할 것입니다.
    
    candidates_x = []
    candidates_y = []
    
    for i in range(128):
        if i in [70, 97, 62]: continue # 제외할 주소
        
        # 획득 직전 프레임에서의 오차 계산
        diff_x = np.abs(ram[pickup_frames, i].astype(int) - sub_x[pickup_frames].astype(int))
        diff_y = np.abs(ram[pickup_frames, i].astype(int) - sub_y[pickup_frames].astype(int))
        
        mean_diff_x = np.mean(diff_x)
        mean_diff_y = np.mean(diff_y)
        
        # 오차가 20 픽셀 이내로 지속적으로 일치하는 주소 찾기
        if mean_diff_x < 20:
            candidates_x.append((i, mean_diff_x))
        if mean_diff_y < 20:
            candidates_y.append((i, mean_diff_y))
            
    print("\n--- 화면 상 잠수부 X 좌표 후보 (Submarine X와 획득 시점에 겹치는 주소) ---")
    for addr, diff in sorted(candidates_x, key=lambda x: x[1]):
        print(f"RAM[{addr}]: 평균 오차 {diff:.1f}")
        
    print("\n--- 화면 상 잠수부 Y 좌표 후보 (Submarine Y와 획득 시점에 겹치는 주소) ---")
    for addr, diff in sorted(candidates_y, key=lambda x: x[1]):
        print(f"RAM[{addr}]: 평균 오차 {diff:.1f}")
        
    print("\n참고: OCAtari / AtariARI 연구에 따르면 적/잠수부 등 화면 상 객체 정보는 보통 RAM[30~34] 근처나 다른 배열에 저장됩니다.")

if __name__ == "__main__":
    main()
