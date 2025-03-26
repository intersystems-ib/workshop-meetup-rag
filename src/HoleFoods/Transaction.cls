/// Use or operation of this code is subject to acceptance of the license available in the code repository for this code.
/// Instances of this class represent specific transactions within
/// the HoleFoods demo.<br/>
/// This table combines both "actual" data and "budget" data so that the sample cube
/// can show comparisons between the two.<br/>
/// Actual records will have a true value for the Actual property.<br/>
/// Budget records will have a false value for the Actual property. The budget value will
/// be stored in the TargetAmount property. Budget values are set for each city, product, and
/// month (using the first day of the month).
Class HoleFoods.Transaction Extends %Persistent [ DdlAllowed, SqlTableName = SalesTransaction ]
{

/// Track changes to this class.
Parameter DSTIME = "auto";

Index DateOfSale On DateOfSale;

Index Product On Product [ Type = bitmap ];

Index Outlet On Outlet [ Type = bitmap ];

/// If true, this represents an actual sale
/// otherwise this represents a sales target.
Property Actual As %Boolean;

/// Date of this sale.
Property DateOfSale As %Date;

/// Product sold.
Property Product As Product;

ForeignKey FKProduct(Product) References Product();

/// Store or other outlet in which the sale occurred.
Property Outlet As Outlet;

/// Channel product was sold through: "Retail" or "Online".
Property Channel As %String(DISPLAYLIST = ",Retail,Online", VALUELIST = ",1,2");

/// Actual amount of this sale.
Property AmountOfSale As %Numeric(SCALE = 2);

/// Units sold.
Property UnitsSold As %Integer;

/// Discount amount.
Property Discount As %Numeric(SCALE = 2);

/// For budget items, this is the target value for a period,region, and product.
Property TargetAmount As %Numeric(SCALE = 2);

/// Customer comment on this transaction (if any).
Property Comment As %String(MAXLEN = 500);

/// US Zipcode of customer (if provided).
Property ZipCode As %String(MAXLEN = 25);

/// Latitude of customer (determined from zip code).
Property Latitude As %Double;

/// Longitude of customer (determined from zip code).
Property Longitude As %Double;

Storage Default
{
<Data name="TransactionDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>Actual</Value>
</Value>
<Value name="3">
<Value>DateOfSale</Value>
</Value>
<Value name="4">
<Value>Product</Value>
</Value>
<Value name="5">
<Value>Outlet</Value>
</Value>
<Value name="6">
<Value>Channel</Value>
</Value>
<Value name="7">
<Value>AmountOfSale</Value>
</Value>
<Value name="8">
<Value>UnitsSold</Value>
</Value>
<Value name="9">
<Value>Discount</Value>
</Value>
<Value name="10">
<Value>TargetAmount</Value>
</Value>
<Value name="11">
<Value>Comment</Value>
</Value>
<Value name="12">
<Value>ZipCode</Value>
</Value>
<Value name="13">
<Value>Latitude</Value>
</Value>
<Value name="14">
<Value>Longitude</Value>
</Value>
</Data>
<DataLocation>^HoleFoods.TransactionD</DataLocation>
<DefaultData>TransactionDefaultData</DefaultData>
<ExtentSize>1909</ExtentSize>
<IdLocation>^HoleFoods.TransactionD</IdLocation>
<IndexLocation>^HoleFoods.TransactionI</IndexLocation>
<Property name="%%CLASSNAME">
<AverageFieldSize>2</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,16,$lb("-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000"),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,21,21),$lb(758198320,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,758198320))</Histogram>
<OutlierSelectivity>.999999:</OutlierSelectivity>
<Selectivity>0.0001%</Selectivity>
</Property>
<Property name="%%ID">
<AverageFieldSize>3.87</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(1,128,256,384,512,639,766,893,1020,1147,1274,1401,1528,1655,1782,1909),$lb(1,0,0,0,0,0,0,0,1,1,1,1,1,1,1),"")</Histogram>
<Selectivity>1</Selectivity>
</Property>
<Property name="Actual">
<AverageFieldSize>2.52</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1),$lb(1,1,1,1,1,1,1,0,1,1,1,1,1,1,1),"")</Histogram>
<Selectivity>50.0000%</Selectivity>
</Property>
<Property name="AmountOfSale">
<AverageFieldSize>3.45</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,8,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,1.76,2.95,3.95,4.95,5.95,11.48,22.95,183.6),$lb(21,21,21,21,21,21,21,0,0,0,0,0,0,0,0),"")</Histogram>
<OutlierSelectivity>.476166:</OutlierSelectivity>
<Selectivity>0.3613%</Selectivity>
</Property>
<Property name="Channel">
<AverageFieldSize>2.47</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" "," "," "," "," "," "," "," "," 1"," 2"," 2"," 2"," 2"," 2"," 2"," 2"),$lb(2,2,2,2,2,2,2,1,1,3,3,3,3,3,3),$lb(536870912,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,822083584,822083584,838860800,0,0,0,0,0,0,0,0,0,0,0,0,540147712))</Histogram>
<Selectivity>33.3333%</Selectivity>
</Property>
<Property name="Comment">
<AverageFieldSize>2.63</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," SHOPPER WAS HAPPY ABOUT RIDICULOUS PRICES"),$lb(2,2,2,2,2,2,2,2,2,2,2,2,2,2,1),$lb(536870912,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1397247824,542328911))</Histogram>
<OutlierSelectivity>.983237:</OutlierSelectivity>
<Selectivity>0.0645%</Selectivity>
</Property>
<Property name="DateOfSale">
<AverageFieldSize>4.8</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(65014,65226,65379,65542,65695,65865,66022,66168,66315,66444,66565,66687,66833,66956,67053,67175),$lb(2,2,2,2,2,1,2,2,2,2,2,2,2,1,2),"")</Histogram>
<OutlierSelectivity>.011524:66534</OutlierSelectivity>
<Selectivity>0.1153%</Selectivity>
</Property>
<Property name="Discount">
<AverageFieldSize>2.4</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,8,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,0,0,0,0,.1,.1,.1,.5),$lb(21,21,21,21,21,21,21,0,1,1,1,0,2,2,1),"")</Histogram>
<Selectivity>20.0000%</Selectivity>
</Property>
<Property name="Latitude">
<AverageFieldSize>2.98</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,14,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,40.24581700000000239,48.36323099999999897),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,0,1),"")</Histogram>
<OutlierSelectivity>.877423:</OutlierSelectivity>
<Selectivity>0.0531%</Selectivity>
</Property>
<Property name="Longitude">
<AverageFieldSize>2.98</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,14,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-86.12981700000000274,-70.07337499999999864),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,1,1),"")</Histogram>
<OutlierSelectivity>.877423:</OutlierSelectivity>
<Selectivity>0.0531%</Selectivity>
</Property>
<Property name="Outlet">
<AverageFieldSize>3.61</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(1,1,2,4,5,6,12,13,17,21,25,27,29,30,34,35),$lb(1,0,0,0,0,0,1,1,0,1,1,1,0,1,1),"")</Histogram>
<Selectivity>2.9412%</Selectivity>
</Property>
<Property name="Product">
<AverageFieldSize>9</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb("SKU-101","SKU-195","SKU-203","SKU-222","SKU-223","SKU-287","SKU-287","SKU-296","SKU-451","SKU-601","SKU-601","SKU-708","SKU-708","SKU-709","SKU-900","SKU-928"),$lb(5,4,5,6,5,8,5,4,4,8,4,8,6,4,5),$lb(1397445933,808517632,959774720,825832704,842019584,808648704,842137600,838860800,855638016,842203136,943128576,0,0,943128576,959840256,842610176,875901184,875901184,909127936,0,0,909127936,925906944,0,0,939524096,956301312,925907200,959459328,808452096,842530816,1397445933))</Histogram>
<Selectivity>5.8824%</Selectivity>
</Property>
<Property name="TargetAmount">
<AverageFieldSize>2.93</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,8,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,1,2.4,3.6,4.8,6,8.4,23,184),$lb(21,21,21,21,21,21,21,0,0,0,0,0,0,0,0),"")</Histogram>
<OutlierSelectivity>.523834:</OutlierSelectivity>
<Selectivity>0.3476%</Selectivity>
</Property>
<Property name="UnitsSold">
<AverageFieldSize>2.52</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,8,$lb(-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,-10000000000000000000,1,1,1,1,1,1,3,8),$lb(21,21,21,21,21,21,21,0,1,1,1,1,1,0,0),"")</Histogram>
<OutlierSelectivity>.476166:</OutlierSelectivity>
<Selectivity>6.5479%</Selectivity>
</Property>
<Property name="ZipCode">
<AverageFieldSize>2.61</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" "," "," "," "," "," "," "," "," "," "," "," "," "," "," 30907"," 98856"),$lb(2,2,2,2,2,2,2,2,2,2,2,2,2,1,1),$lb(536870912,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,858798384,858798384,959985717,540620856))</Histogram>
<OutlierSelectivity>.877423:</OutlierSelectivity>
<Selectivity>0.0531%</Selectivity>
</Property>
<SQLMap name="$Transaction">
<BlockCount>-4</BlockCount>
</SQLMap>
<SQLMap name="DateOfSale">
<BlockCount>-16</BlockCount>
</SQLMap>
<SQLMap name="IDKEY">
<BlockCount>-56</BlockCount>
</SQLMap>
<SQLMap name="Outlet">
<BlockCount>-8</BlockCount>
</SQLMap>
<SQLMap name="Product">
<BlockCount>-4</BlockCount>
</SQLMap>
<StreamLocation>^HoleFoods.TransactionS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
