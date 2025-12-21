import requests
from bs4 import BeautifulSoup
import json
import time
import re


class IUEScraper:
    def __init__(self):
        self.base_url = "https://ects.ieu.edu.tr/new/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        self.all_courses = {}

    def get_course_links(self, dept_url):
        print(f"\n--- Sayfa inceleniyor: {dept_url}")
        try:
            response = requests.get(dept_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            # syllabus linklerini yakala
            for a in soup.find_all('a', href=re.compile(r'syllabus')):
                href = a.get('href', '')
                full_link = href if href.startswith('http') else self.base_url + href
                if full_link not in links:
                    links.append(full_link)
            return links
        except Exception as e:
            print(f"HATA: Linkler alınamadı: {e}")
            return []

    def scrape_course_details(self, url, department_name):
        try:
            if 'lang=en' not in url:
                url += '&lang=en'

            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Veri iskeleti
            course_data = {
                "department": department_name,
                "course_code": "",
                "course_name": "",
                "semester": "",
                "type": "",
                "ects": "",
                "objectives": "",
                "description": "",
                "prerequisites": "None",
                "weekly_topics": []
            }

            # TÜM tablo hücrelerini (td ve th) tara
            cells = soup.find_all(['td', 'th'])
            for i, cell in enumerate(cells):
                text = cell.get_text(strip=True)

                # Eğer hücrede anahtar kelime varsa, bir sonraki hücre veridir
                if i + 1 < len(cells):
                    next_val = cells[i + 1].get_text(strip=True)

                    if text == "Code":
                        course_data["course_code"] = next_val
                    elif text == "Course Name":
                        course_data["course_name"] = next_val
                    elif text == "Semester":
                        course_data["semester"] = next_val
                    elif text == "ECTS":
                        course_data["ects"] = next_val
                    elif text == "Course Type":
                        course_data["type"] = next_val
                    elif "Course Objectives" in text:
                        course_data["objectives"] = next_val
                    elif "Course Description" in text:
                        course_data["description"] = next_val
                    elif "Prerequisites" in text:
                        course_data["prerequisites"] = next_val

            # Haftalık Konular (Tablo ID'si ile çekme)
            weeks_table = soup.find('table', id='weeks')
            if weeks_table:
                for row in weeks_table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        course_data["weekly_topics"].append({
                            "week": cols[0].get_text(strip=True),
                            "topic": cols[1].get_text(strip=True)
                        })

            return course_data
        except Exception as e:
            print(f"HATA: {url} taranırken hata: {e}")
            return None

    def run(self, departments):
        for dept_name, url in departments.items():
            course_links = self.get_course_links(url)
            print(f"--- {dept_name}: {len(course_links)} link bulundu. İşleniyor...")

            for link in course_links:
                # Linkten ders kodunu tahmin et (Örn: SE 380)
                code_match = re.search(r'course_code=([^&]+)', link)
                if code_match:
                    temp_code = code_match.group(1).replace('+', ' ').replace('%20', ' ')

                    # ORTAK DERS KONTROLÜ
                    if temp_code in self.all_courses:
                        continue

                    details = self.scrape_course_details(link, dept_name)
                    if details and details["course_code"]:
                        # Sayfadan gelen gerçek kodu kullan
                        real_code = details["course_code"]
                        self.all_courses[real_code] = details
                        print(f"Başarılı: {real_code}")

                time.sleep(0.1)  # Hızlandırdım

        # Kayıt
        with open('iue_curriculum.json', 'w', encoding='utf-8') as f:
            json.dump(list(self.all_courses.values()), f, ensure_ascii=False, indent=4)
        print(f"\nBİTTİ! Toplam {len(self.all_courses)} benzersiz ders 'iue_curriculum.json' dosyasına yazıldı.")


if __name__ == "__main__":
    target_depts = {
        "Software Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=se.cs.ieu.edu.tr&sid=curr",
        "Computer Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=ce.cs.ieu.edu.tr&sid=curr",
        "Industrial Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=is.cs.ieu.edu.tr&sid=curr",
        "Electrical and Electronics Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=ete.cs.ieu.edu.tr&sid=curr"
    }

    scraper = IUEScraper()
    scraper.run(target_depts)