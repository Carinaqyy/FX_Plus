###############################################################################
# Copyright [Carina Quan]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
class StringToFileGenerator:
    def __init__(self, file_name='output.txt'):
        self.file_name = file_name

    def generate_file(self, string):
        try:
            with open(self.file_name, 'w') as file:
                file.write(f"{string}\n")
            print(f"File '{self.file_name}' successfully generated.")
        except Exception as e:
            print(f"Error generating file: {e}")

