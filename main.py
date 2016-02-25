import urllib.parse, urllib.request
import getpass
import ssl
import captcha
import re
import time
import random
import settings
from pyquery import PyQuery as pq

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import cookielib
except Exception as e:
    import http.cookiejar as cookielib

def SettingUp():
    Cookie = cookielib.LWPCookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(Cookie))
    urllib.request.install_opener(opener)

def GetHeaders():
    header = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',
                'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding':'deflate,sdch',
                'Accept-Language':'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4'}
    return header

def Post(url, data):
    req = urllib.request.Request(url = url, data = urllib.parse.urlencode(data).encode('utf-8'), headers = GetHeaders())
    req.add_header('Content-Type', 'application/x-www-form-urlencoded');
    try:
        response = urllib.request.urlopen(req)
    except urllib.request.URLError as e:
        print(e)
        exit()
    return response
def Get(url, data = None):
    if data:
        url += '?' + urllib.parse.urlencode(data)
    req = urllib.request.Request(url = url, headers = GetHeaders())
    try:
        response = urllib.request.urlopen(req)
    except urllib.request.URLError as e:
        print(e)
        exit()
    return response

SettingUp()

captchacache = captcha.CaptchaCache("Captcha.txt")
captchacache.load()

options = settings.Settings('Settings.txt')
options.load()

username = options.get('username')
password = options.get('password')

def login(ask = True):
    Get('http://zhjwxk.cic.tsinghua.edu.cn/xklogin.do')

    image = Get('http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1').read()
    imagefile = open('captcha.jpeg', 'wb')
    imagefile.write(image)
    imagefile.close()

    #username = input('Username: ')
    #password = getpass.getpass()

    res = captchacache.search(image)
    if res:
        code = res
    elif ask:
        code = input('Captcha: ')
        code = code.upper()
    else:
        return False

    data = {'j_username': username,
            'j_password': password,
            'captchaflag': 'login1',
            '_login_image_': code}

    res = Post('https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do', data)
    if res.status != 200:
        print('ERROR.')
        exit()

    content = res.read().decode('gb2312')
    error = re.findall(r'<div align="center">(.*?)</div>', content, re.S)

    if len(error)!=0:
        print('ERROR: ' + error[0])
        exit()
    else:
        print('Login Success!')
        captchacache.add(image, code)
        captchacache.save()
        return True

def continueLogin():
    i = 0
    while not login(False) and i < 100:
        i += 1
    if i == 100:
        exit()

continueLogin()

url = 'http://zhjwxk.cic.tsinghua.edu.cn/xkBks.vxkBksXkbBs.do'

res = Get(url , { 'm': 'rxSearch', 'p_xnxq': '2015-2016-2'})
content = res.read().decode('gb2312')
d = pq(content)

data = {i.name: '' if i.value == None else i.value
    for i in d(':input') if i.attrib.get('type') != 'reset'
    and i.attrib.get('type') != 'button'}
data['page'] = '-1'
data['m'] = 'rxSearch'
data['is_zyrxk'] = ''
data['p_sort.asc1'] = 'true'
data['p_sort.asc2'] = 'true'
data['tokenPriFlag'] = 'rx'

data['p_kch'] = '00640312'

content = ''
res = None

ntry = 0
nhit = 0

def run():
    global content, res, ntry, nhit
    while True:
        res = Post(url, data)
        content = res.read().decode('gb2312')
        s = pq(content)("table:eq(2) tr")
        if len(s) == 0:
            continueLogin()
        for i in s:
            res = pq(i)("td:eq(4) span")[0].text
            if res != '0':
                data['p_rx_id'] = pq(i)("td input")[0].attrib['value']
                data['goPageNumber'] = '1'
                data['m'] = 'saveRxKc'
                res = Post(url, data)
                nhit += 1
        ntry += 1
        print('Try: ' + str(ntry) + '\tHit: ' + str(nhit))
        num = random.normalvariate(2,2)
        while num <= 0.1:
            num = random.normalvariate(2,2)
        time.sleep(num)
        
    
    
