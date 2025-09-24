import requests

url = "http://152.136.168.63:11996/audio/speech"
data = {
    "input": "普京称将回应中国对俄试行免签 【普京称将回应中国对俄试行免签】财联社9月5日电，据CCTV国际时讯报道，俄罗斯总统普京9月4日表示，中国将对俄罗斯公民试行免签政策意义重大",  # 文本
    "voice": "44_1d5fadf24e00d759",  # 对应预注册的角色
    "model": "index-tts"  # 模型名（可任意填写，项目暂不校验）
}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("openai_style_output.wav", "wb") as f:
        f.write(response.content)