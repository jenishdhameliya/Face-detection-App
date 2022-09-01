from django import forms


class Memberform(forms.Form):
    name = forms.CharField(label="Member Name",max_length=250)
    age = forms.IntegerField(label="Age of the Member")