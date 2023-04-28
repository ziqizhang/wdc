
import urllib.request, urllib.error, urllib.parse

url = 'https://search.electoralcommission.org.uk/Search/Spending?currentPage=1&rows=20&query=Newsquest%20Media%20Group&sort=TotalExpenditure&order=desc&tab=1&et=pp&includeOutsideSection75=true&evt=ukparliament&ev=3696&optCols=ExpenseCategoryName&optCols=AmountInEngland&optCols=AmountInScotland&optCols=AmountInWales&optCols=AmountInNorthernIreland&optCols=DatePaid'

response = urllib.request.urlopen(url)
webContent = response.read().decode('UTF-8')

print(webContent)