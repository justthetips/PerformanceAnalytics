from setuptools import setup

setup(
    name='PerformanceAnalytics',
    version='0.12a',
    packages=['tests', 'tests.performanceanalytics', 'performanceanalytics', 'performanceanalytics.table',
              'performanceanalytics.charts'],
    url='https://github.com/justthetips/PerformanceAnalytics',
    license='MIT',
    author='Jacob Bourne',
    author_email='jacob.bourne@gmail.com',
    description='A port of R\'s performance analytics to python'
)
